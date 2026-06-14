from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, Dict, Any, List

try:
    from .config import ModelConfig
except ImportError:
    from config import ModelConfig


class ConflictDetector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.entity_dim = config.entity_embedding_dim
        self.attribute_dim = config.attribute_embedding_dim
        self.threshold = config.conflict_threshold

        self.entity_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.entity_dim),
            nn.LayerNorm(self.entity_dim),
            nn.Tanh()
        )

        self.attribute_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attribute_dim),
            nn.LayerNorm(self.attribute_dim),
            nn.Tanh()
        )

        self.conflict_scorer = nn.Sequential(
            nn.Linear(self.attribute_dim * 2 + 1, self.attribute_dim),
            nn.ReLU(),
            nn.Linear(self.attribute_dim, 1),
            nn.Sigmoid()
        )

        self.temporal_decay = nn.Parameter(torch.tensor(0.01))

    def compute_entity_similarity(
        self,
        entity_embeds: torch.Tensor,
        memory_entity_embeds: torch.Tensor,
    ) -> torch.Tensor:
        entity_embeds_norm = F.normalize(entity_embeds, p=2, dim=-1)
        memory_entity_embeds_norm = F.normalize(memory_entity_embeds, p=2, dim=-1)
        similarity = torch.einsum("bqd,md->bmq", entity_embeds_norm, memory_entity_embeds_norm)
        return similarity

    def compute_attribute_conflict(
        self,
        attribute_embeds: torch.Tensor,
        memory_attribute_embeds: torch.Tensor,
        timestamps: torch.Tensor,
        memory_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = attribute_embeds.shape
        memory_size = memory_attribute_embeds.shape[0]

        attr_diff = attribute_embeds.unsqueeze(1) - memory_attribute_embeds.unsqueeze(0).unsqueeze(2)
        attr_dist = torch.norm(attr_diff, p=2, dim=-1)

        time_diff = timestamps.unsqueeze(1).float() - memory_timestamps.unsqueeze(0).unsqueeze(2).float()
        time_mask = time_diff > 0
        time_diff = time_diff.abs()

        temporal_factor = torch.exp(-self.temporal_decay * time_diff)
        temporal_factor = temporal_factor * time_mask.float()

        scorer_input = torch.cat([
            attribute_embeds.unsqueeze(1).expand(-1, memory_size, -1, -1),
            memory_attribute_embeds.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, seq_len, -1),
            attr_dist.unsqueeze(-1)
        ], dim=-1)

        conflict_scores = self.conflict_scorer(scorer_input).squeeze(-1)
        conflict_scores = conflict_scores * temporal_factor

        return conflict_scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        memory_entity_embeds: torch.Tensor,
        memory_attribute_embeds: torch.Tensor,
        memory_timestamps: torch.Tensor,
        memory_valid_mask: torch.Tensor,
        current_timestamps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        batch_size, seq_len, _ = hidden_states.shape

        entity_embeds = self.entity_encoder(hidden_states)
        attribute_embeds = self.attribute_encoder(hidden_states)

        entity_sim = self.compute_entity_similarity(entity_embeds, memory_entity_embeds)
        entity_sim = entity_sim.masked_fill(~memory_valid_mask.unsqueeze(-1), float('-inf'))

        top_entity_sim, top_entity_indices = torch.topk(
            entity_sim, k=min(16, memory_valid_mask.sum().item()), dim=1
        )

        conflict_scores = self.compute_attribute_conflict(
            attribute_embeds,
            memory_attribute_embeds,
            current_timestamps,
            memory_timestamps,
        )

        batch_idx = torch.arange(batch_size).view(-1, 1, 1)
        seq_idx = torch.arange(seq_len).view(1, 1, -1)
        top_conflict_scores = conflict_scores[batch_idx, top_entity_indices, seq_idx]

        conflicts = top_conflict_scores > self.threshold
        conflict_positions = conflicts.any(dim=1)

        conflict_details = {
            "entity_similarity": top_entity_sim,
            "top_entity_indices": top_entity_indices,
            "conflict_scores": top_conflict_scores,
            "conflict_mask": conflicts,
            "entity_embeds": entity_embeds,
            "attribute_embeds": attribute_embeds,
        }

        return conflict_positions, top_entity_indices, conflict_details


class GradientGatedRewriter(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.entity_dim = config.entity_embedding_dim
        self.attribute_dim = config.attribute_embedding_dim
        self.temperature = config.gradient_gate_temperature
        self.enable_fusion = config.enable_fusion
        self.backprop_window = config.backprop_window

        self.retention_gate = nn.Sequential(
            nn.Linear(self.attribute_dim * 2 + self.entity_dim + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(self.attribute_dim * 2 + self.entity_dim + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.attribute_dim),
            nn.Sigmoid()
        )

        self.value_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.gradient_scaler = nn.Parameter(torch.tensor(0.1))
        self.representation_corrector = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

    def compute_gates(
        self,
        new_entity_embeds: torch.Tensor,
        new_attribute_embeds: torch.Tensor,
        old_attribute_embeds: torch.Tensor,
        new_timestamps: torch.Tensor,
        old_timestamps: torch.Tensor,
        conflict_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_conflicts, seq_len = conflict_scores.shape

        time_diff = new_timestamps.unsqueeze(1).float() - old_timestamps.float()
        time_diff_norm = torch.tanh(time_diff.abs() / 1000.0)

        gate_input = torch.cat([
            new_entity_embeds.unsqueeze(1).expand(-1, num_conflicts, -1, -1),
            new_attribute_embeds.unsqueeze(1).expand(-1, num_conflicts, -1, -1),
            old_attribute_embeds,
            time_diff_norm.unsqueeze(-1),
            conflict_scores.unsqueeze(-1),
        ], dim=-1)

        retention_gate = self.retention_gate(gate_input).squeeze(-1)
        retention_gate = F.gumbel_softmax(
            torch.stack([retention_gate, 1 - retention_gate], dim=-1),
            tau=self.temperature,
            hard=False,
            dim=-1
        )[..., 0]

        fusion_gate = self.fusion_gate(gate_input) if self.enable_fusion else torch.ones_like(gate_input[..., :self.attribute_dim])

        return retention_gate, fusion_gate

    def fuse_values(
        self,
        new_values: torch.Tensor,
        old_values: torch.Tensor,
        retention_gate: torch.Tensor,
        fusion_gate: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_conflicts, seq_len, _ = old_values.shape

        new_expanded = new_values.unsqueeze(1).expand(-1, num_conflicts, -1, -1)

        if self.enable_fusion:
            fusion_gate_mean = fusion_gate.mean(dim=-1, keepdim=True)
            fusion_gate_expanded = fusion_gate_mean.expand(-1, -1, -1, self.hidden_dim)
            fused_values = fusion_gate_expanded * new_expanded + (1 - fusion_gate_expanded) * old_values
        else:
            fused_values = new_expanded

        retention_gate_expanded = retention_gate.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
        final_values = retention_gate_expanded * old_values + (1 - retention_gate_expanded) * fused_values

        return final_values

    def correct_representations(
        self,
        hidden_states: torch.Tensor,
        conflict_positions: torch.Tensor,
        memory_indices: torch.Tensor,
        memory_values: torch.Tensor,
        current_timestamps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        correction_signals = torch.zeros_like(hidden_states)
        gradient_masks = torch.zeros(batch_size, seq_len, device=hidden_states.device)

        for b in range(batch_size):
            conflict_idx = torch.where(conflict_positions[b])[0]
            if len(conflict_idx) == 0:
                continue

            for c_idx in conflict_idx:
                current_time = current_timestamps[b, c_idx].item()
                window_start = max(0, c_idx - self.backprop_window)
                window_length = c_idx - window_start + 1

                if window_length <= 0:
                    continue

                related_indices = memory_indices[b, :, c_idx]
                related_indices = related_indices[related_indices >= 0]

                if len(related_indices) > 0:
                    related_values = memory_values[related_indices]
                    context = hidden_states[b, window_start:c_idx+1].unsqueeze(0)

                    corrected, _ = self.representation_corrector(context)
                    correction = corrected - context

                    alpha = self.gradient_scaler * torch.sigmoid(
                        -torch.arange(window_length, device=hidden_states.device).float() / (self.backprop_window / 4)
                    ).view(1, -1, 1)

                    correction_signals[b, window_start:c_idx+1] += (alpha * correction).squeeze(0)
                    gradient_masks[b, window_start:c_idx+1] = torch.maximum(
                        gradient_masks[b, window_start:c_idx+1],
                        alpha.squeeze(-1).squeeze(0)
                    )

        return correction_signals, gradient_masks

    def forward(
        self,
        hidden_states: torch.Tensor,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        new_entity_embeds: torch.Tensor,
        new_attribute_embeds: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        memory_entity_embeds: torch.Tensor,
        memory_attribute_embeds: torch.Tensor,
        memory_timestamps: torch.Tensor,
        conflict_indices: torch.Tensor,
        conflict_scores: torch.Tensor,
        current_timestamps: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        num_conflicts = conflict_indices.shape[1]

        batch_idx = torch.arange(batch_size).view(-1, 1, 1)
        seq_idx = torch.arange(seq_len).view(1, 1, -1)

        old_attribute_embeds = memory_attribute_embeds[conflict_indices]
        old_values = memory_values[conflict_indices]
        old_timestamps = memory_timestamps[conflict_indices]

        retention_gate, fusion_gate = self.compute_gates(
            new_entity_embeds,
            new_attribute_embeds,
            old_attribute_embeds,
            current_timestamps,
            old_timestamps,
            conflict_scores,
        )

        fused_values = self.fuse_values(
            new_values,
            old_values,
            retention_gate,
            fusion_gate,
        )

        fused_attribute_embeds = fusion_gate * new_attribute_embeds.unsqueeze(1).expand(-1, num_conflicts, -1, -1) + \
            (1 - fusion_gate) * old_attribute_embeds

        final_attribute_embeds = retention_gate.unsqueeze(-1) * old_attribute_embeds + \
            (1 - retention_gate.unsqueeze(-1)) * fused_attribute_embeds

        correction_signals, gradient_masks = self.correct_representations(
            hidden_states,
            conflict_scores.max(dim=1)[0] > self.config.conflict_threshold,
            conflict_indices,
            memory_values,
            current_timestamps,
        )

        rewrite_info = {
            "memory_indices": conflict_indices,
            "retention_gate": retention_gate,
            "fusion_gate": fusion_gate,
            "new_values": fused_values,
            "new_attribute_embeds": final_attribute_embeds,
            "update_mask": (1 - retention_gate).detach(),
        }

        return rewrite_info, correction_signals, gradient_masks
