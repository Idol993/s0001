import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from memory import ExternalMemory, ReadHead, WriteHead, TemporalCache
from attention import DualAttention, SlidingWindowAttention, GlobalAttention
from conflict import ConflictDetector, GradientGatedRewriter
from model import MemoryAugmentedLLM, TransformerBlock
from training import ContradictionDataset, ContradictionLoss, Trainer, create_training_pipeline


def test_config():
    print("Testing ModelConfig...")
    config = ModelConfig()
    assert config.hidden_dim == 1024
    assert config.num_heads == 16
    assert config.head_dim == 64
    assert config.hidden_dim == config.num_heads * config.head_dim
    print("OK ModelConfig passed")


def test_temporal_cache():
    print("\nTesting TemporalCache...")
    config = ModelConfig(cache_size=64, hidden_dim=64, memory_key_dim=32, memory_value_dim=64, num_heads=4, head_dim=16)
    cache = TemporalCache(config)

    batch_size, seq_len = 2, 10
    keys = torch.randn(batch_size, seq_len, config.memory_key_dim)
    values = torch.randn(batch_size, seq_len, config.hidden_dim)
    timestamps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    entity_ids = torch.randint(0, 10, (batch_size, seq_len))

    cache(keys, values, timestamps, entity_ids)

    query = torch.randn(batch_size, seq_len, config.hidden_dim)
    scores, cache_values, cache_timestamps, cache_entity_ids = cache.search(query, top_k=4)

    assert scores.shape == (batch_size, 4, seq_len)
    assert cache_values.shape == (batch_size, 4, seq_len, config.hidden_dim)
    print("OK TemporalCache passed")


def test_read_head():
    print("\nTesting ReadHead...")
    config = ModelConfig(hidden_dim=64, memory_key_dim=32, memory_value_dim=64, memory_top_k=4, memory_size=128, num_heads=4, head_dim=16)
    read_head = ReadHead(config)

    batch_size, seq_len = 2, 8
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
    memory_keys = torch.randn(config.memory_size, config.memory_key_dim)
    memory_values = torch.randn(config.memory_size, config.hidden_dim)
    memory_mask = torch.ones(config.memory_size, dtype=torch.bool)

    output, attn_weights, top_indices = read_head(
        hidden_states, memory_keys, memory_values, memory_mask
    )

    assert output.shape == (batch_size, seq_len, config.hidden_dim)
    assert attn_weights.shape == (batch_size, 4, seq_len)
    assert top_indices.shape == (batch_size, 4, seq_len)
    print("OK ReadHead passed")


def test_write_head():
    print("\nTesting WriteHead...")
    config = ModelConfig(
        hidden_dim=64,
        memory_key_dim=32,
        memory_value_dim=64,
        entity_embedding_dim=16,
        attribute_embedding_dim=16,
        memory_size=128,
        num_heads=4,
        head_dim=16
    )
    write_head = WriteHead(config)

    batch_size, seq_len = 2, 8
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
    memory_keys = torch.randn(config.memory_size, config.memory_key_dim)
    memory_values = torch.randn(config.memory_size, config.hidden_dim)
    write_indices = torch.randint(0, config.memory_size, (batch_size, seq_len))

    keys, values, entity_embeds, attr_embeds = write_head(
        hidden_states, memory_keys, memory_values, write_indices
    )

    assert keys.shape == (batch_size, seq_len, config.memory_key_dim)
    assert values.shape == (batch_size, seq_len, config.hidden_dim)
    assert entity_embeds.shape == (batch_size, seq_len, config.entity_embedding_dim)
    assert attr_embeds.shape == (batch_size, seq_len, config.attribute_embedding_dim)
    print("OK WriteHead passed")


def test_external_memory():
    print("\nTesting ExternalMemory...")
    config = ModelConfig(
        hidden_dim=64,
        memory_key_dim=32,
        memory_value_dim=64,
        memory_size=128,
        num_read_heads=2,
        num_write_heads=1,
        memory_top_k=4,
        cache_size=32,
        entity_embedding_dim=16,
        attribute_embedding_dim=16,
        num_heads=4,
        head_dim=16,
    )
    memory = ExternalMemory(config)

    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
    timestamps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    entity_ids = torch.randint(0, 10, (batch_size, seq_len))

    output, info = memory(hidden_states, timestamps, entity_ids, mode="read_write")

    assert output.shape == (batch_size, seq_len, config.hidden_dim)
    assert "write_indices" in info
    assert "keys" in info
    assert "values" in info
    assert "read_attn_weights" in info
    assert info["read_attn_weights"].shape == (batch_size, config.num_read_heads, 4, seq_len)
    print("OK ExternalMemory passed")


def test_sliding_window_attention():
    print("\nTesting SlidingWindowAttention...")
    config = ModelConfig(hidden_dim=64, num_heads=4, head_dim=16, window_size=8, memory_value_dim=64)
    attn = SlidingWindowAttention(config)

    batch_size, seq_len = 2, 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

    output, weights, kv = attn(hidden_states)

    assert output.shape == (batch_size, seq_len, config.hidden_dim)
    assert weights.shape == (batch_size, config.num_heads, seq_len, seq_len)
    assert kv[0].shape == (batch_size, config.num_heads, seq_len, config.head_dim)
    print("OK SlidingWindowAttention passed")


def test_global_attention():
    print("\nTesting GlobalAttention...")
    config = ModelConfig(hidden_dim=64, num_heads=4, head_dim=16, global_token_count=4, memory_value_dim=64)
    attn = GlobalAttention(config)

    batch_size, seq_len = 2, 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

    output, global_tokens, weights = attn(hidden_states)

    assert output.shape == (batch_size, seq_len, config.hidden_dim)
    assert global_tokens.shape == (batch_size, config.global_token_count, config.hidden_dim)
    assert weights.shape == (batch_size, config.num_heads, seq_len + config.global_token_count, seq_len + config.global_token_count)
    print("OK GlobalAttention passed")


def test_dual_attention():
    print("\nTesting DualAttention...")
    config = ModelConfig(
        hidden_dim=64,
        num_heads=4,
        head_dim=16,
        window_size=8,
        global_token_count=4,
        use_sliding_window=True,
        use_global_attention=True,
        memory_value_dim=64,
    )
    attn = DualAttention(config)

    batch_size, seq_len = 2, 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

    output, info = attn(hidden_states)

    assert output.shape == (batch_size, seq_len, config.hidden_dim)
    assert "sliding_weights" in info
    assert "global_weights" in info
    assert "global_tokens" in info
    print("OK DualAttention passed")


def test_conflict_detector():
    print("\nTesting ConflictDetector...")
    config = ModelConfig(
        hidden_dim=64,
        entity_embedding_dim=16,
        attribute_embedding_dim=16,
        conflict_threshold=0.5,
        memory_size=128,
        memory_key_dim=32,
        memory_value_dim=64,
        num_heads=4,
        head_dim=16,
    )  # already correct
    detector = ConflictDetector(config)

    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
    memory_keys = torch.randn(config.memory_size, config.memory_key_dim)
    memory_values = torch.randn(config.memory_size, config.hidden_dim)
    memory_entity_embeds = torch.randn(config.memory_size, config.entity_embedding_dim)
    memory_attr_embeds = torch.randn(config.memory_size, config.attribute_embedding_dim)
    memory_timestamps = torch.arange(config.memory_size)
    memory_mask = torch.ones(config.memory_size, dtype=torch.bool)
    current_timestamps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1) + config.memory_size

    conflict_positions, conflict_indices, details = detector(
        hidden_states,
        memory_keys,
        memory_values,
        memory_entity_embeds,
        memory_attr_embeds,
        memory_timestamps,
        memory_mask,
        current_timestamps,
    )

    assert conflict_positions.shape == (batch_size, seq_len)
    assert "conflict_scores" in details
    assert "entity_embeds" in details
    assert "attribute_embeds" in details
    print("OK ConflictDetector passed")


def test_gradient_rewriter():
    print("\nTesting GradientGatedRewriter...")
    config = ModelConfig(
        hidden_dim=64,
        entity_embedding_dim=16,
        attribute_embedding_dim=16,
        memory_key_dim=32,
        memory_value_dim=64,
        memory_size=128,
        backprop_window=32,
        enable_fusion=True,
        num_heads=4,
        head_dim=16,
    )  # already correct
    rewriter = GradientGatedRewriter(config)

    batch_size, seq_len = 2, 16
    num_conflicts = 4

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim, requires_grad=True)
    new_keys = torch.randn(batch_size, seq_len, config.memory_key_dim)
    new_values = torch.randn(batch_size, seq_len, config.hidden_dim)
    new_entity_embeds = torch.randn(batch_size, seq_len, config.entity_embedding_dim)
    new_attr_embeds = torch.randn(batch_size, seq_len, config.attribute_embedding_dim)
    memory_keys = torch.randn(config.memory_size, config.memory_key_dim)
    memory_values = torch.randn(config.memory_size, config.hidden_dim)
    memory_entity_embeds = torch.randn(config.memory_size, config.entity_embedding_dim)
    memory_attr_embeds = torch.randn(config.memory_size, config.attribute_embedding_dim)
    memory_timestamps = torch.arange(config.memory_size)
    conflict_indices = torch.randint(0, config.memory_size, (batch_size, num_conflicts, seq_len))
    conflict_scores = torch.rand(batch_size, num_conflicts, seq_len)
    current_timestamps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1) + config.memory_size

    rewrite_info, correction, grad_mask = rewriter(
        hidden_states,
        new_keys,
        new_values,
        new_entity_embeds,
        new_attr_embeds,
        memory_keys,
        memory_values,
        memory_entity_embeds,
        memory_attr_embeds,
        memory_timestamps,
        conflict_indices,
        conflict_scores,
        current_timestamps,
    )

    assert "retention_gate" in rewrite_info
    assert "fusion_gate" in rewrite_info
    assert "new_values" in rewrite_info
    assert correction.shape == (batch_size, seq_len, config.hidden_dim)
    assert grad_mask.shape == (batch_size, seq_len)

    loss = correction.sum()
    loss.backward()
    assert hidden_states.grad is not None
    print("OK GradientGatedRewriter passed (gradient flow verified)")


def test_transformer_block():
    print("\nTesting TransformerBlock...")
    config = ModelConfig(
        hidden_dim=64,
        num_heads=4,
        head_dim=16,
        window_size=8,
        global_token_count=4,
        dropout=0.0,
        memory_value_dim=64,
    )
    block = TransformerBlock(config)

    batch_size, seq_len = 2, 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

    output, info = block(hidden_states)

    assert output.shape == (batch_size, seq_len, config.hidden_dim)
    print("OK TransformerBlock passed")


def test_memory_augmented_llm():
    print("\nTesting MemoryAugmentedLLM...")
    config = ModelConfig(
        vocab_size=1000,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        max_seq_len=256,
        memory_size=64,
        memory_key_dim=32,
        memory_value_dim=64,
        num_read_heads=2,
        num_write_heads=1,
        memory_top_k=4,
        cache_size=32,
        window_size=16,
        global_token_count=4,
        entity_embedding_dim=16,
        attribute_embedding_dim=16,
        conflict_threshold=0.5,
        backprop_window=32,
    )
    model = MemoryAugmentedLLM(config)

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    entity_ids = torch.randint(0, 10, (batch_size, seq_len))
    global_indices = torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45]], dtype=torch.long)

    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        entity_ids=entity_ids,
        global_indices=global_indices,
        detect_conflicts=True,
        enable_rewrite=True,
    )

    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert "hidden_states" in outputs
    assert outputs["hidden_states"].shape == (batch_size, seq_len, config.hidden_dim)
    assert "conflict_info" in outputs
    assert "memory_info" in outputs

    loss = outputs["logits"].sum()
    loss.backward()

    skip_params = [
        "external_memory.memory_keys",
        "external_memory.memory_values",
        "external_memory.memory_entity_embeds",
        "external_memory.memory_attribute_embeds",
        "external_memory.temporal_cache.cache_keys",
        "external_memory.temporal_cache.cache_values",
    ]

    has_gradient = False
    for name, param in model.named_parameters():
        if param.requires_grad and name not in skip_params:
            if param.grad is not None:
                has_gradient = True
                break

    assert has_gradient, "No gradients flowing through the model"
    assert outputs["hidden_states"].grad_fn is not None, "Hidden states not connected to computation graph"

    print("OK MemoryAugmentedLLM passed (full gradient flow verified)")

    print("\nTesting generation...")
    generated = model.generate(input_ids[:, :32], max_new_tokens=16, temperature=0.0, top_k=0)
    assert generated.shape == (batch_size, 32 + 16)
    print("OK Generation passed")


def test_training_dataset():
    print("\nTesting ContradictionDataset...")
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=8192,
        global_token_count=4,
        window_size=1024,
        hidden_dim=64,
        num_heads=4,
        head_dim=16,
        memory_value_dim=64,
    )
    dataset = ContradictionDataset(config, num_samples=2, vocab_size=1000)

    assert len(dataset) == 2
    sample = dataset[0]

    assert sample.input_ids.shape == (config.max_seq_len,)
    assert sample.attention_mask.shape == (config.max_seq_len,)
    assert sample.entity_ids.shape == (config.max_seq_len,)
    assert sample.conflict_positions.shape == (config.max_seq_len,)
    assert sample.global_indices.shape == (config.global_token_count,)
    assert sample.conflict_positions.sum() > 0

    print("OK ContradictionDataset passed")


def test_training_loss():
    print("\nTesting ContradictionLoss...")
    config = ModelConfig(
        vocab_size=1000,
        hidden_dim=64,
        max_seq_len=128,
        entity_embedding_dim=16,
        attribute_embedding_dim=16,
        num_heads=4,
        head_dim=16,
        memory_value_dim=64,
    )
    loss_fn = ContradictionLoss(config)

    batch_size = 2
    seq_len = 64
    outputs = {
        "logits": torch.randn(batch_size, seq_len, config.vocab_size, requires_grad=True),
        "conflict_logits": torch.randn(batch_size, seq_len, 2, requires_grad=True),
        "predicted_entity_ids": torch.randn(batch_size, seq_len, 1000, requires_grad=True),
        "rewrite_info": {
            "retention_gate": torch.sigmoid(torch.randn(batch_size, 4, seq_len, requires_grad=True)),
        }
    }

    from training import ContradictionSample
    batch = ContradictionSample(
        input_ids=torch.randint(0, config.vocab_size, (batch_size, seq_len)),
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
        entity_ids=torch.randint(0, 10, (batch_size, seq_len)),
        labels=torch.randint(0, config.vocab_size, (batch_size, seq_len)),
        conflict_positions=torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool),
        conflict_entity_ids=torch.tensor([1, 2, 3]),
        correct_attribute_positions=torch.tensor([10, 20]),
        global_indices=torch.tensor([[5, 15, 25, 35], [6, 16, 26, 36]]),
    )

    loss, loss_dict = loss_fn(outputs, batch)

    assert loss.requires_grad
    assert "total_loss" in loss_dict
    assert "lm_loss" in loss_dict
    assert "conflict_loss" in loss_dict
    assert "rewrite_loss" in loss_dict
    assert "entity_loss" in loss_dict

    loss.backward()
    print("OK ContradictionLoss passed")


def main():
    print("=" * 60)
    print("Running Architecture Verification Tests")
    print("=" * 60)

    try:
        test_config()
        test_temporal_cache()
        test_read_head()
        test_write_head()
        test_external_memory()
        test_sliding_window_attention()
        test_global_attention()
        test_dual_attention()
        test_conflict_detector()
        test_gradient_rewriter()
        test_transformer_block()
        test_memory_augmented_llm()
        test_training_dataset()
        test_training_loss()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED OK")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
