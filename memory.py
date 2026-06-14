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


class TemporalCache(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.cache_size = config.cache_size
        self.hidden_dim = config.hidden_dim
        self.key_dim = config.memory_key_dim

        self.query_proj = nn.Linear(self.hidden_dim, self.key_dim)
        self.cache_keys = nn.Parameter(torch.zeros(self.cache_size, self.key_dim))
        self.cache_values = nn.Parameter(torch.zeros(self.cache_size, self.hidden_dim))
        self.cache_timestamps = nn.Parameter(torch.zeros(self.cache_size, dtype=torch.long), requires_grad=False)
        self.cache_entity_ids = nn.Parameter(torch.zeros(self.cache_size, dtype=torch.long), requires_grad=False)

        self.write_ptr = nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.cache_keys)
        nn.init.xavier_uniform_(self.cache_values)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        timestamps: torch.Tensor,
        entity_ids: torch.Tensor,
    ) -> None:
        batch_size = keys.shape[0]
        seq_len = keys.shape[1]

        for b in range(batch_size):
            for t in range(seq_len):
                idx = self.write_ptr.item()
                self.cache_keys.data[idx] = keys[b, t]
                self.cache_values.data[idx] = values[b, t]
                self.cache_timestamps.data[idx] = timestamps[b, t]
                self.cache_entity_ids.data[idx] = entity_ids[b, t]
                self.write_ptr.data = (self.write_ptr.data + 1) % self.cache_size

    def search(self, query: torch.Tensor, top_k: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query_proj = self.query_proj(query)
        scores = torch.einsum("bqd,cd->bcq", query_proj, self.cache_keys)
        scores = scores / (self.key_dim ** 0.5)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, self.cache_size), dim=1)

        batch_size = query.shape[0]
        seq_len = query.shape[1]

        top_values = self.cache_values[top_indices]
        top_timestamps = self.cache_timestamps[top_indices]
        top_entity_ids = self.cache_entity_ids[top_indices]

        return top_scores, top_values, top_timestamps, top_entity_ids

    def get_entity_entries(self, entity_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self.cache_entity_ids == entity_id
        if not mask.any():
            return None, None, None

        keys = self.cache_keys[mask]
        values = self.cache_values[mask]
        timestamps = self.cache_timestamps[mask]
        return keys, values, timestamps


class ReadHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.key_dim = config.memory_key_dim
        self.hidden_dim = config.hidden_dim
        self.top_k = config.memory_top_k

        self.query_proj = nn.Linear(self.hidden_dim, self.key_dim)
        self.content_gate = nn.Sequential(
            nn.Linear(self.hidden_dim + self.key_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        self.output_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        memory_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        query = self.query_proj(hidden_states)

        scores = torch.einsum("bqd,md->bmq", query, memory_keys)
        scores = scores / (self.key_dim ** 0.5)

        if memory_valid_mask is not None:
            memory_valid_mask = memory_valid_mask.unsqueeze(-1)
            scores = scores.masked_fill(~memory_valid_mask, float('-inf'))

        top_scores, top_indices = torch.topk(scores, k=min(self.top_k, memory_keys.shape[0]), dim=1)
        attn_weights = F.softmax(top_scores, dim=1)

        top_values = memory_values[top_indices]
        read_values = torch.einsum("bmq,bmqd->bqd", attn_weights, top_values)

        content_query = torch.cat([hidden_states, query], dim=-1)
        gate = self.content_gate(content_query)
        output = gate * read_values + (1 - gate) * hidden_states
        output = self.output_proj(output)

        return output, attn_weights, top_indices


class WriteHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.key_dim = config.memory_key_dim
        self.hidden_dim = config.hidden_dim

        self.key_proj = nn.Linear(self.hidden_dim, self.key_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.entity_proj = nn.Linear(self.hidden_dim, config.entity_embedding_dim)
        self.attribute_proj = nn.Linear(self.hidden_dim, config.attribute_embedding_dim)

        self.write_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        self.erase_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.key_dim),
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        write_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = self.key_proj(hidden_states)
        values = self.value_proj(hidden_states)
        entity_embeds = self.entity_proj(hidden_states)
        attribute_embeds = self.attribute_proj(hidden_states)

        write_gate = self.write_gate(hidden_states)
        values = write_gate * values + (1 - write_gate) * memory_values[write_indices]

        erase_gate = self.erase_gate(hidden_states)
        keys = erase_gate * keys + (1 - erase_gate) * memory_keys[write_indices]

        return keys, values, entity_embeds, attribute_embeds


class ExternalMemory(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.memory_size = config.memory_size
        self.key_dim = config.memory_key_dim
        self.value_dim = config.memory_value_dim
        self.hidden_dim = config.hidden_dim

        self.memory_keys = nn.Parameter(torch.zeros(self.memory_size, self.key_dim))
        self.memory_values = nn.Parameter(torch.zeros(self.memory_size, self.value_dim))
        self.memory_entity_embeds = nn.Parameter(torch.zeros(self.memory_size, config.entity_embedding_dim))
        self.memory_attribute_embeds = nn.Parameter(torch.zeros(self.memory_size, config.attribute_embedding_dim))
        self.memory_timestamps = nn.Parameter(torch.zeros(self.memory_size, dtype=torch.long), requires_grad=False)
        self.memory_valid = nn.Parameter(torch.zeros(self.memory_size, dtype=torch.bool), requires_grad=False)
        self.memory_entity_ids = nn.Parameter(torch.full((self.memory_size,), -1, dtype=torch.long), requires_grad=False)

        self.read_heads = nn.ModuleList([ReadHead(config) for _ in range(config.num_read_heads)])
        self.write_heads = nn.ModuleList([WriteHead(config) for _ in range(config.num_write_heads)])

        self.temporal_cache = TemporalCache(config)

        self.write_ptr = nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)
        self.read_head_fusion = nn.Linear(config.num_read_heads * self.hidden_dim, self.hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.memory_keys)
        nn.init.xavier_uniform_(self.memory_values)
        nn.init.xavier_uniform_(self.memory_entity_embeds)
        nn.init.xavier_uniform_(self.memory_attribute_embeds)

    def _get_write_indices(self, batch_size: int, seq_len: int) -> torch.Tensor:
        indices = []
        for _ in range(batch_size):
            start = self.write_ptr.item()
            batch_indices = [(start + i) % self.memory_size for i in range(seq_len)]
            indices.append(batch_indices)
            self.write_ptr.data = (self.write_ptr.data + seq_len) % self.memory_size
        return torch.tensor(indices, dtype=torch.long)

    def read(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, seq_len, _ = hidden_states.shape

        read_outputs = []
        all_attn_weights = []
        all_top_indices = []

        for read_head in self.read_heads:
            output, attn_weights, top_indices = read_head(
                hidden_states,
                self.memory_keys,
                self.memory_values,
                self.memory_valid,
            )
            read_outputs.append(output)
            all_attn_weights.append(attn_weights)
            all_top_indices.append(top_indices)

        fused_read = torch.cat(read_outputs, dim=-1)
        fused_read = self.read_head_fusion(fused_read)

        cache_scores, cache_values, cache_timestamps, cache_entity_ids = self.temporal_cache.search(
            hidden_states, top_k=self.config.memory_top_k
        )
        cache_attn = F.softmax(cache_scores, dim=1)
        cache_read = torch.einsum("bmq,bmqd->bqd", cache_attn, cache_values)

        output = fused_read + cache_read

        return output, {
            "read_attn_weights": torch.stack(all_attn_weights, dim=1),
            "read_top_indices": torch.stack(all_top_indices, dim=1),
            "cache_read": cache_read,
        }

    def write(
        self,
        hidden_states: torch.Tensor,
        timestamps: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        batch_size, seq_len, _ = hidden_states.shape

        write_indices = self._get_write_indices(batch_size, seq_len)

        all_keys = []
        all_values = []
        all_entity_embeds = []
        all_attribute_embeds = []

        for write_head in self.write_heads:
            keys, values, entity_embeds, attribute_embeds = write_head(
                hidden_states,
                self.memory_keys,
                self.memory_values,
                write_indices,
            )
            all_keys.append(keys)
            all_values.append(values)
            all_entity_embeds.append(entity_embeds)
            all_attribute_embeds.append(attribute_embeds)

        keys = torch.stack(all_keys, dim=2).mean(dim=2)
        values = torch.stack(all_values, dim=2).mean(dim=2)
        entity_embeds = torch.stack(all_entity_embeds, dim=2).mean(dim=2)
        attribute_embeds = torch.stack(all_attribute_embeds, dim=2).mean(dim=2)

        for b in range(batch_size):
            for t in range(seq_len):
                idx = write_indices[b, t].item()
                self.memory_keys.data[idx] = keys[b, t].detach()
                self.memory_values.data[idx] = values[b, t].detach()
                self.memory_entity_embeds.data[idx] = entity_embeds[b, t].detach()
                self.memory_attribute_embeds.data[idx] = attribute_embeds[b, t].detach()
                self.memory_timestamps.data[idx] = timestamps[b, t].item()
                self.memory_valid.data[idx] = True
                if entity_ids is not None:
                    self.memory_entity_ids.data[idx] = entity_ids[b, t].item()

        if entity_ids is None:
            entity_ids = torch.zeros_like(timestamps)

        self.temporal_cache(keys.detach(), values.detach(), timestamps.detach(), entity_ids.detach())

        return {
            "write_indices": write_indices,
            "keys": keys,
            "values": values,
            "entity_embeds": entity_embeds,
            "attribute_embeds": attribute_embeds,
        }

    def rewrite(
        self,
        memory_indices: torch.Tensor,
        new_keys: Optional[torch.Tensor] = None,
        new_values: Optional[torch.Tensor] = None,
        new_entity_embeds: Optional[torch.Tensor] = None,
        new_attribute_embeds: Optional[torch.Tensor] = None,
        update_mask: Optional[torch.Tensor] = None,
    ) -> None:
        for idx in memory_indices.flatten():
            idx = idx.item()
            if idx < 0 or idx >= self.memory_size:
                continue

            if update_mask is not None:
                mask = update_mask.flatten()[memory_indices.flatten() == idx]
            else:
                mask = 1.0

            if new_keys is not None:
                self.memory_keys.data[idx] = (
                    mask * new_keys.flatten(0, 1)[memory_indices.flatten() == idx] +
                    (1 - mask) * self.memory_keys.data[idx]
                )
            if new_values is not None:
                self.memory_values.data[idx] = (
                    mask * new_values.flatten(0, 1)[memory_indices.flatten() == idx] +
                    (1 - mask) * self.memory_values.data[idx]
                )
            if new_entity_embeds is not None:
                self.memory_entity_embeds.data[idx] = (
                    mask * new_entity_embeds.flatten(0, 1)[memory_indices.flatten() == idx] +
                    (1 - mask) * self.memory_entity_embeds.data[idx]
                )
            if new_attribute_embeds is not None:
                self.memory_attribute_embeds.data[idx] = (
                    mask * new_attribute_embeds.flatten(0, 1)[memory_indices.flatten() == idx] +
                    (1 - mask) * self.memory_attribute_embeds.data[idx]
                )

    def get_entity_memory(self, entity_id: int) -> Dict[str, torch.Tensor]:
        mask = self.memory_entity_ids == entity_id
        if not mask.any():
            return {}

        return {
            "keys": self.memory_keys[mask],
            "values": self.memory_values[mask],
            "entity_embeds": self.memory_entity_embeds[mask],
            "attribute_embeds": self.memory_attribute_embeds[mask],
            "timestamps": self.memory_timestamps[mask],
            "indices": torch.where(mask)[0],
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestamps: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None,
        mode: str = "read_write",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert mode in ["read", "write", "read_write"]

        write_info = {}
        if mode in ["write", "read_write"]:
            write_info = self.write(hidden_states, timestamps, entity_ids)

        read_output = hidden_states
        read_info = {}
        if mode in ["read", "read_write"]:
            read_output, read_info = self.read(hidden_states)

        return read_output, {**write_info, **read_info}
