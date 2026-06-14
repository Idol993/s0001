from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, Dict, Any

try:
    from .config import ModelConfig
except ImportError:
    from config import ModelConfig


class SlidingWindowAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

    def _create_sliding_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        mask = torch.triu(mask, diagonal=1) | torch.tril(mask, diagonal=-self.window_size)
        return ~mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)

        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        full_seq_len = k.shape[2]

        attn_weights = torch.einsum("b h q d, b h k d -> b h q k", q, k)
        attn_weights = attn_weights / (self.head_dim ** 0.5)

        sliding_mask = self._create_sliding_mask(full_seq_len, hidden_states.device)
        sliding_mask = sliding_mask[-seq_len:, :]

        if attention_mask is not None:
            if past_key_values is not None:
                past_len = past_key_values[0].shape[2]
                full_mask = torch.ones(
                    batch_size, full_seq_len, device=hidden_states.device, dtype=torch.bool
                )
                full_mask[:, past_len:] = attention_mask
                attention_mask = full_mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attn_weights = attn_weights.masked_fill(~attention_mask, float('-inf'))

        attn_weights = attn_weights.masked_fill(~sliding_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum("b h q k, b h k d -> b h q d", attn_weights, v)
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, (k, v)


class GlobalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.global_token_count = config.global_token_count

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.global_tokens = nn.Parameter(torch.zeros(self.global_token_count, self.hidden_dim))
        self.global_token_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.global_tokens)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        global_tokens = repeat(self.global_tokens, "g d -> b g d", b=batch_size)

        if global_indices is not None:
            global_hidden = hidden_states.gather(
                1, global_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            )
            gate = self.global_token_gate(global_hidden)
            global_tokens = gate * global_hidden + (1 - gate) * global_tokens

        combined_states = torch.cat([global_tokens, hidden_states], dim=1)
        combined_seq_len = combined_states.shape[1]

        q = self.q_proj(combined_states)
        k = self.k_proj(combined_states)
        v = self.v_proj(combined_states)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)

        attn_weights = torch.einsum("b h q d, b h k d -> b h q k", q, k)
        attn_weights = attn_weights / (self.head_dim ** 0.5)

        causal_mask = torch.ones(combined_seq_len, combined_seq_len, dtype=torch.bool, device=hidden_states.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)

        global_mask = torch.ones_like(causal_mask)
        global_mask[:self.global_token_count, :] = True
        global_mask[:, :self.global_token_count] = True
        global_mask = global_mask & ~causal_mask

        local_mask = ~causal_mask
        local_mask[:, :self.global_token_count] = False

        full_mask = global_mask | local_mask
        attn_weights = attn_weights.masked_fill(~full_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if attention_mask is not None:
            full_attention_mask = torch.ones(
                batch_size, combined_seq_len, device=hidden_states.device, dtype=torch.bool
            )
            full_attention_mask[:, self.global_token_count:] = attention_mask
            full_attention_mask = full_attention_mask.unsqueeze(1).unsqueeze(1)
            attn_weights = attn_weights.masked_fill(~full_attention_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum("b h q k, b h k d -> b h q d", attn_weights, v)
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")
        attn_output = self.o_proj(attn_output)

        global_output = attn_output[:, :self.global_token_count, :]
        regular_output = attn_output[:, self.global_token_count:, :]

        return regular_output, global_output, attn_weights


class DualAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        self.use_sliding_window = config.use_sliding_window
        self.use_global_attention = config.use_global_attention

        if self.use_sliding_window:
            self.sliding_attn = SlidingWindowAttention(config)

        if self.use_global_attention:
            self.global_attn = GlobalAttention(config)

        self.fusion_gate = nn.Sequential(
            nn.Linear(3 * self.hidden_dim if self.use_sliding_window and self.use_global_attention else self.hidden_dim,
                      self.hidden_dim),
            nn.Sigmoid()
        )

        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        global_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, seq_len, _ = hidden_states.shape

        outputs = []
        info = {}

        if self.use_sliding_window:
            sliding_output, sliding_weights, new_kv = self.sliding_attn(
                hidden_states, attention_mask, past_key_values
            )
            outputs.append(sliding_output)
            info["sliding_weights"] = sliding_weights
            info["past_key_values"] = new_kv

        if self.use_global_attention:
            global_output, global_tokens, global_weights = self.global_attn(
                hidden_states, attention_mask, global_indices
            )
            outputs.append(global_output)
            info["global_weights"] = global_weights
            info["global_tokens"] = global_tokens

        if len(outputs) == 1:
            output = outputs[0]
        else:
            combined = torch.cat([hidden_states, outputs[0], outputs[1]], dim=-1)
            gate = self.fusion_gate(combined)
            output = gate * outputs[0] + (1 - gate) * outputs[1]

        output = self.dropout(output)
        output = self.output_norm(output + hidden_states)

        return output, info
