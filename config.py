from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    hidden_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    head_dim: int = 64
    max_seq_len: int = 100000
    dropout: float = 0.1

    memory_size: int = 4096
    memory_key_dim: int = 128
    memory_value_dim: int = 1024
    num_read_heads: int = 4
    num_write_heads: int = 2
    memory_top_k: int = 8
    cache_size: int = 512

    window_size: int = 4096
    global_token_count: int = 32
    use_sliding_window: bool = True
    use_global_attention: bool = True

    conflict_threshold: float = 0.75
    gradient_gate_temperature: float = 0.1
    enable_fusion: bool = True
    backprop_window: int = 1024

    entity_embedding_dim: int = 256
    attribute_embedding_dim: int = 256

    def __post_init__(self):
        assert self.hidden_dim == self.num_heads * self.head_dim
        assert self.memory_value_dim == self.hidden_dim
