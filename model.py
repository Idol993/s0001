from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, Dict, Any, List

try:
    from .config import ModelConfig
    from .memory import ExternalMemory
    from .attention import DualAttention
    from .conflict import ConflictDetector, GradientGatedRewriter
except ImportError:
    from config import ModelConfig
    from memory import ExternalMemory
    from attention import DualAttention
    from conflict import ConflictDetector, GradientGatedRewriter


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        self.dual_attn = DualAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
            nn.Dropout(config.dropout)
        )
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        global_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        attn_output, attn_info = self.dual_attn(
            self.norm1(hidden_states),
            attention_mask,
            past_key_values,
            global_indices,
        )
        hidden_states = hidden_states + self.dropout(attn_output)

        ff_output = self.feed_forward(self.norm2(hidden_states))
        hidden_states = hidden_states + self.dropout(ff_output)

        return hidden_states, attn_info


class MemoryAugmentedLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.num_layers = config.num_layers

        self.token_embedding = nn.Embedding(config.vocab_size, self.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, self.hidden_dim)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

        self.external_memory = ExternalMemory(config)
        self.conflict_detector = ConflictDetector(config)
        self.gradient_rewriter = GradientGatedRewriter(config)

        self.memory_integration_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )

        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, config.vocab_size, bias=False)

        self.entity_id_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1000)
        )

        self.conflict_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)
        )

        self.dropout = nn.Dropout(config.dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        entity_ids: Optional[torch.Tensor] = None,
        global_indices: Optional[torch.Tensor] = None,
        memory_mode: str = "read_write",
        detect_conflicts: bool = True,
        enable_rewrite: bool = True,
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        timestamps = positions.clone()

        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(positions)
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        past_key_values = None
        all_attn_info = []

        for layer in self.layers:
            hidden_states, attn_info = layer(
                hidden_states,
                attention_mask,
                past_key_values,
                global_indices,
            )
            past_key_values = attn_info.get("past_key_values")
            all_attn_info.append(attn_info)

        entity_logits = self.entity_id_predictor(hidden_states)
        predicted_entity_ids = entity_logits.argmax(dim=-1)
        if entity_ids is None:
            entity_ids = predicted_entity_ids

        memory_output, memory_info = self.external_memory(
            hidden_states,
            timestamps,
            entity_ids,
            mode=memory_mode,
        )

        memory_gate_input = torch.cat([hidden_states, memory_output], dim=-1)
        memory_gate = self.memory_integration_gate(memory_gate_input)
        hidden_states = memory_gate * memory_output + (1 - memory_gate) * hidden_states

        conflict_info = {}
        rewrite_info = {}
        correction_signals = torch.zeros_like(hidden_states)
        gradient_masks = torch.zeros(batch_size, seq_len, device=device)

        if detect_conflicts and memory_info.get("keys") is not None:
            conflict_positions, conflict_indices, conflict_details = self.conflict_detector(
                hidden_states,
                self.external_memory.memory_keys,
                self.external_memory.memory_values,
                self.external_memory.memory_entity_embeds,
                self.external_memory.memory_attribute_embeds,
                self.external_memory.memory_timestamps,
                self.external_memory.memory_valid,
                timestamps,
            )

            conflict_info = {
                "conflict_positions": conflict_positions,
                "conflict_indices": conflict_indices,
                **conflict_details,
            }

            if enable_rewrite and conflict_positions.any():
                rewrite_result, correction_signals, gradient_masks = self.gradient_rewriter(
                    hidden_states,
                    memory_info.get("keys", torch.zeros_like(hidden_states[..., :self.config.memory_key_dim])),
                    memory_info.get("values", hidden_states),
                    memory_info.get("entity_embeds", torch.zeros_like(hidden_states[..., :self.config.entity_embedding_dim])),
                    memory_info.get("attribute_embeds", torch.zeros_like(hidden_states[..., :self.config.attribute_embedding_dim])),
                    self.external_memory.memory_keys,
                    self.external_memory.memory_values,
                    self.external_memory.memory_entity_embeds,
                    self.external_memory.memory_attribute_embeds,
                    self.external_memory.memory_timestamps,
                    conflict_indices,
                    conflict_details["conflict_scores"],
                    timestamps,
                )

                num_updates = self.external_memory.rewrite(
                    rewrite_result["memory_indices"],
                    rewrite_result["new_values"],
                    rewrite_result["new_attribute_embeds"],
                    rewrite_result["update_mask"],
                )
                rewrite_result["num_memory_updates"] = num_updates
                rewrite_info = rewrite_result

        hidden_states = hidden_states + correction_signals

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        conflict_classifier_input = torch.cat([
            hidden_states,
            memory_output,
        ], dim=-1)
        conflict_logits = self.conflict_classifier(conflict_classifier_input)

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "memory_output": memory_output,
            "memory_info": memory_info,
            "conflict_info": conflict_info,
            "rewrite_info": rewrite_info,
            "correction_signals": correction_signals,
            "gradient_masks": gradient_masks,
            "entity_logits": entity_logits,
            "predicted_entity_ids": predicted_entity_ids,
            "conflict_logits": conflict_logits,
            "attention_info": all_attn_info,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        self.eval()
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.forward(
                    generated[:, -self.config.window_size:],
                    detect_conflicts=False,
                    enable_rewrite=False,
                )
                next_token_logits = outputs["logits"][:, -1, :]

                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, k=top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                if temperature == 0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    next_token_probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(next_token_probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

        return generated
