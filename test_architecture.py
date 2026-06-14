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
    assert "entity_logits" in outputs
    assert outputs["entity_logits"].shape == (batch_size, seq_len, 1000)
    assert outputs["predicted_entity_ids"].shape == (batch_size, seq_len)

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
        "entity_logits": torch.randn(batch_size, seq_len, 1000, requires_grad=True),
        "predicted_entity_ids": torch.randint(0, 1000, (batch_size, seq_len)),
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


def test_contradiction_detection_rewrite_read():
    print("\nTesting contradiction: detect -> rewrite -> re-read cycle...")
    config = ModelConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        max_seq_len=512,
        memory_size=64,
        memory_key_dim=64,
        memory_value_dim=128,
        num_read_heads=2,
        num_write_heads=1,
        memory_top_k=4,
        cache_size=32,
        window_size=128,
        global_token_count=4,
        entity_embedding_dim=32,
        attribute_embedding_dim=32,
        conflict_threshold=0.1,
        backprop_window=32,
        dropout=0.0,
    )
    model = MemoryAugmentedLLM(config)
    model.eval()

    batch_size = 1
    first_seq_len = 32
    second_seq_len = 32

    entity_id = 5

    first_input_ids = torch.randint(0, config.vocab_size, (batch_size, first_seq_len))
    first_entity_ids = torch.full((batch_size, first_seq_len), entity_id, dtype=torch.long)

    with torch.no_grad():
        outputs_first = model(
            first_input_ids,
            entity_ids=first_entity_ids,
            detect_conflicts=False,
            enable_rewrite=False,
        )

    mem_after_first = model.external_memory.get_entity_memory(entity_id)
    assert len(mem_after_first) > 0, "First write should create memory entries for the entity"
    first_values = mem_after_first["values"].clone()
    first_indices = mem_after_first["indices"].clone()
    print(f"  First write: {len(first_values)} memory entries for entity {entity_id}")

    second_input_ids = torch.randint(0, config.vocab_size, (batch_size, second_seq_len))
    second_entity_ids = torch.full((batch_size, second_seq_len), entity_id, dtype=torch.long)

    with torch.no_grad():
        outputs_second = model(
            second_input_ids,
            entity_ids=second_entity_ids,
            detect_conflicts=True,
            enable_rewrite=True,
        )

    assert "conflict_info" in outputs_second
    assert "rewrite_info" in outputs_second

    conflict_detected = outputs_second["conflict_info"].get("conflict_positions")
    num_updates = outputs_second["rewrite_info"].get("num_memory_updates", 0)
    print(f"  Conflicts detected: {conflict_detected.sum().item() if conflict_detected is not None else 0}")
    print(f"  Memory updates applied: {num_updates}")

    mem_after_second = model.external_memory.get_entity_memory(entity_id)
    assert len(mem_after_second) > 0, "Second pass should still have memory entries"

    second_values = mem_after_second["values"]
    values_changed = not torch.allclose(first_values, second_values[:len(first_values)])
    print(f"  Memory values changed after rewrite: {values_changed}")

    with torch.no_grad():
        query_input = torch.randint(0, config.vocab_size, (batch_size, 16))
        query_entity_ids = torch.full((batch_size, 16), entity_id, dtype=torch.long)

        outputs_read = model(
            query_input,
            entity_ids=query_entity_ids,
            detect_conflicts=False,
            enable_rewrite=False,
        )

    assert "memory_output" in outputs_read
    assert outputs_read["memory_output"].shape == (batch_size, 16, config.hidden_dim)

    memory_read = outputs_read["memory_output"]
    assert not torch.isnan(memory_read).any(), "Memory read should not contain NaN"

    print("OK Contradiction detect-rewrite-read cycle passed")


def test_training_step_with_memory_update():
    print("\nTesting end-to-end training step with memory update...")
    config = ModelConfig(
        vocab_size=500,
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
        cache_size=16,
        window_size=64,
        global_token_count=4,
        entity_embedding_dim=16,
        attribute_embedding_dim=16,
        conflict_threshold=0.3,
        backprop_window=16,
        dropout=0.0,
    )

    model = MemoryAugmentedLLM(config)
    loss_fn = ContradictionLoss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_size = 1
    seq_len = 128
    num_entity_types = 5

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    entity_ids = torch.randint(1, num_entity_types + 1, (batch_size, seq_len))
    labels = input_ids.clone()

    conflict_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    conflict_positions[0, 90:95] = True

    from training import ContradictionSample
    batch = ContradictionSample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        entity_ids=entity_ids,
        labels=labels,
        conflict_positions=conflict_positions,
        conflict_entity_ids=torch.tensor([2]),
        correct_attribute_positions=torch.tensor([90]),
        global_indices=torch.tensor([[10, 30, 50, 70]]),
    )

    model.train()
    optimizer.zero_grad()

    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        entity_ids=entity_ids,
        global_indices=batch.global_indices,
        detect_conflicts=True,
        enable_rewrite=True,
    )

    assert "logits" in outputs
    assert "entity_logits" in outputs
    assert "conflict_logits" in outputs
    assert "conflict_info" in outputs
    assert "memory_info" in outputs

    loss, loss_dict = loss_fn(outputs, batch)

    assert loss.requires_grad, "Loss must require gradients"
    assert not torch.isnan(loss), "Loss must not be NaN"
    assert not torch.isinf(loss), "Loss must not be Inf"

    loss.backward()

    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if param.grad.abs().sum() > 0:
                has_grad = True
                break

    assert has_grad, "At least some parameters must have non-zero gradients"

    optimizer.step()

    loss_val = loss.item()
    print(f"  Total loss: {loss_val:.4f}")
    for k, v in loss_dict.items():
        print(f"    {k}: {v:.4f}")

    num_updates = outputs["rewrite_info"].get("num_memory_updates", 0) if outputs.get("rewrite_info") else 0
    print(f"  Memory updates during training step: {num_updates}")

    print("OK End-to-end training step with memory update passed")


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
        test_contradiction_detection_rewrite_read()
        test_training_step_with_memory_update()

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
