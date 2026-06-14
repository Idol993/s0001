from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import random
import math
from dataclasses import dataclass

try:
    from .config import ModelConfig
    from .model import MemoryAugmentedLLM
except ImportError:
    from config import ModelConfig
    from model import MemoryAugmentedLLM


@dataclass
class ContradictionSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    entity_ids: torch.Tensor
    labels: torch.Tensor
    conflict_positions: torch.Tensor
    conflict_entity_ids: torch.Tensor
    correct_attribute_positions: torch.Tensor
    global_indices: torch.Tensor


class ContradictionDataset(Dataset):
    def __init__(
        self,
        config: ModelConfig,
        num_samples: int = 100,
        vocab_size: int = 50257,
        seed: int = 42,
    ):
        self.config = config
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_seq_len = config.max_seq_len
        self.window_size = config.window_size
        self.global_token_count = config.global_token_count

        self.entities = [f"实体{i}" for i in range(100)]
        self.attributes = ["身高", "年龄", "体重", "职业", "国籍", "学历", "收入", "爱好"]
        self.attribute_values = {
            "身高": ["170cm", "175cm", "180cm", "185cm", "190cm"],
            "年龄": ["25岁", "30岁", "35岁", "40岁", "45岁"],
            "体重": ["60kg", "65kg", "70kg", "75kg", "80kg"],
            "职业": ["工程师", "教师", "医生", "律师", "艺术家"],
            "国籍": ["中国", "美国", "日本", "德国", "法国"],
            "学历": ["本科", "硕士", "博士", "高中", "大专"],
            "收入": ["10万", "20万", "30万", "50万", "100万"],
            "爱好": ["阅读", "运动", "音乐", "旅行", "摄影"],
        }

        random.seed(seed)
        self.samples = self._generate_samples()

    def _generate_sentence(self, entity: str, attribute: str, value: str) -> str:
        templates = [
            f"{entity}的{attribute}是{value}。",
            f"据了解，{entity}{attribute}为{value}。",
            f"{entity}，其{attribute}是{value}。",
            f"记录显示{entity}的{attribute}为{value}。",
            f"{entity}自我介绍{attribute}是{value}。",
        ]
        return random.choice(templates)

    def _tokenize(self, text: str) -> List[int]:
        tokens = []
        for char in text:
            tokens.append(ord(char) % (self.vocab_size - 10) + 10)
        return tokens

    def _generate_samples(self) -> List[ContradictionSample]:
        samples = []

        for sample_idx in range(self.num_samples):
            seq_len = self.max_seq_len
            num_entities = random.randint(5, 15)
            selected_entities = random.sample(self.entities, num_entities)

            tokens = [1]
            entity_ids = [0]
            attention_mask = [1]

            entity_memory = {}
            conflict_positions = []
            conflict_entity_ids_list = []
            correct_attribute_positions = []

            num_contradictions = random.randint(2, 5)
            contradiction_events = []

            for _ in range(num_contradictions):
                entity = random.choice(selected_entities)
                attribute = random.choice(self.attributes)
                values = random.sample(self.attribute_values[attribute], 2)

                first_pos = random.randint(100, seq_len // 3)
                min_gap = min(40000, seq_len // 4)
                max_gap = min(60000, seq_len // 2)
                second_pos = random.randint(first_pos + min_gap, min(first_pos + max_gap, seq_len - 500))

                contradiction_events.append({
                    "entity": entity,
                    "attribute": attribute,
                    "first_value": values[0],
                    "second_value": values[1],
                    "first_pos": first_pos,
                    "second_pos": second_pos,
                })

                if entity not in entity_memory:
                    entity_memory[entity] = len(entity_memory) + 1

            event_ptr = 0
            contradiction_events.sort(key=lambda x: x["first_pos"])
            processed_second = set()

            while len(tokens) < seq_len:
                current_pos = len(tokens)
                written_event = False

                for i, ev in enumerate(contradiction_events):
                    if i in processed_second:
                        continue
                    if current_pos >= ev["second_pos"] and current_pos < ev["second_pos"] + 50:
                        sentence = self._generate_sentence(
                            ev["entity"], ev["attribute"], ev["second_value"]
                        )
                        sent_tokens = self._tokenize(sentence)
                        tokens.extend(sent_tokens)
                        entity_ids.extend([entity_memory[ev["entity"]]] * len(sent_tokens))
                        attention_mask.extend([1] * len(sent_tokens))

                        conflict_start = len(tokens) - len(sent_tokens)
                        conflict_positions.extend(range(conflict_start, len(tokens)))
                        conflict_entity_ids_list.extend([entity_memory[ev["entity"]]] * len(sent_tokens))
                        correct_attribute_positions.append(conflict_start)
                        processed_second.add(i)
                        written_event = True
                        break

                if not written_event and event_ptr < len(contradiction_events):
                    event = contradiction_events[event_ptr]
                    if current_pos >= event["first_pos"] and current_pos < event["first_pos"] + 50:
                        sentence = self._generate_sentence(
                            event["entity"], event["attribute"], event["first_value"]
                        )
                        sent_tokens = self._tokenize(sentence)
                        tokens.extend(sent_tokens)
                        entity_ids.extend([entity_memory[event["entity"]]] * len(sent_tokens))
                        attention_mask.extend([1] * len(sent_tokens))
                        event_ptr += 1
                        written_event = True

                if not written_event:
                    step = min(32, seq_len - len(tokens))
                    random_tokens = [random.randint(10, self.vocab_size - 1) for _ in range(step)]
                    tokens.extend(random_tokens)
                    entity_ids.extend([0] * step)
                    attention_mask.extend([1] * step)

            tokens = tokens[:seq_len]
            entity_ids = entity_ids[:seq_len]
            attention_mask = attention_mask[:seq_len]

            labels = tokens.copy()
            for pos in correct_attribute_positions:
                if pos < len(labels):
                    labels[pos] = tokens[pos]

            global_indices = torch.tensor(
                sorted(random.sample(range(100, seq_len - 100), self.global_token_count)),
                dtype=torch.long
            )

            conflict_positions_tensor = torch.zeros(seq_len, dtype=torch.bool)
            for pos in conflict_positions:
                if pos < seq_len:
                    conflict_positions_tensor[pos] = True

            samples.append(ContradictionSample(
                input_ids=torch.tensor(tokens, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask, dtype=torch.bool),
                entity_ids=torch.tensor(entity_ids, dtype=torch.long),
                labels=torch.tensor(labels, dtype=torch.long),
                conflict_positions=conflict_positions_tensor,
                conflict_entity_ids=torch.tensor(conflict_entity_ids_list, dtype=torch.long),
                correct_attribute_positions=torch.tensor(correct_attribute_positions, dtype=torch.long),
                global_indices=global_indices,
            ))

        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> ContradictionSample:
        return self.samples[idx]


class ContradictionLoss(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.lm_loss_weight = 1.0
        self.conflict_detection_weight = 0.5
        self.rewrite_loss_weight = 0.3
        self.entity_pred_weight = 0.2

    def forward(
        self,
        outputs: Dict[str, Any],
        batch: ContradictionSample,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = batch.input_ids.shape[0]

        lm_loss = F.cross_entropy(
            outputs["logits"].view(-1, self.config.vocab_size),
            batch.labels.view(-1),
            ignore_index=-100
        )

        conflict_labels = batch.conflict_positions.long()
        conflict_loss = F.cross_entropy(
            outputs["conflict_logits"].view(-1, 2),
            conflict_labels.view(-1)
        )

        entity_loss = F.cross_entropy(
            outputs["entity_logits"].view(-1, 1000),
            batch.entity_ids.view(-1),
            ignore_index=0
        )

        rewrite_loss = torch.tensor(0.0, device=outputs["logits"].device)
        if outputs.get("rewrite_info") and outputs["rewrite_info"]:
            retention_gate = outputs["rewrite_info"]["retention_gate"]
            if retention_gate.numel() > 0:
                target_retention = torch.zeros_like(retention_gate)
                rewrite_loss = F.binary_cross_entropy(retention_gate, target_retention)

        total_loss = (
            self.lm_loss_weight * lm_loss +
            self.conflict_detection_weight * conflict_loss +
            self.rewrite_loss_weight * rewrite_loss +
            self.entity_pred_weight * entity_loss
        )

        loss_dict = {
            "total_loss": total_loss.item(),
            "lm_loss": lm_loss.item(),
            "conflict_loss": conflict_loss.item(),
            "rewrite_loss": rewrite_loss.item(),
            "entity_loss": entity_loss.item(),
        }

        return total_loss, loss_dict


class Trainer:
    def __init__(
        self,
        config: ModelConfig,
        model: MemoryAugmentedLLM,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.loss_fn = ContradictionLoss(config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.gradient_accumulation_steps = 4
        self.max_grad_norm = 1.0
        self.global_step = 0

    def train_step(
        self,
        batch: ContradictionSample,
    ) -> Dict[str, float]:
        self.model.train()

        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        entity_ids = batch.entity_ids.to(self.device)
        labels = batch.labels.to(self.device)
        global_indices = batch.global_indices.to(self.device)

        chunk_size = self.config.window_size * 2
        seq_len = input_ids.shape[1]
        num_chunks = math.ceil(seq_len / chunk_size)

        total_loss = 0.0
        all_loss_dict = {}

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, seq_len)

            chunk_input = input_ids[:, start:end]
            chunk_mask = attention_mask[:, start:end]
            chunk_entity = entity_ids[:, start:end]
            chunk_labels = labels[:, start:end]
            chunk_global = global_indices.clamp(0, end - start - 1)

            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(
                    chunk_input,
                    attention_mask=chunk_mask,
                    entity_ids=chunk_entity,
                    global_indices=chunk_global,
                    detect_conflicts=True,
                    enable_rewrite=True,
                )

                chunk_batch = ContradictionSample(
                    input_ids=chunk_input,
                    attention_mask=chunk_mask,
                    entity_ids=chunk_entity,
                    labels=chunk_labels,
                    conflict_positions=batch.conflict_positions[:, start:end].to(self.device),
                    conflict_entity_ids=batch.conflict_entity_ids.to(self.device),
                    correct_attribute_positions=batch.correct_attribute_positions.to(self.device),
                    global_indices=chunk_global,
                )

                loss, loss_dict = self.loss_fn(outputs, chunk_batch)
                loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward(retain_graph=(chunk_idx < num_chunks - 1))

            total_loss += loss.item() * self.gradient_accumulation_steps
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v / num_chunks

            if outputs.get("gradient_masks") is not None and outputs["gradient_masks"].numel() > 0:
                gradient_mask = outputs["gradient_masks"].to(self.device)
                for name, param in self.model.named_parameters():
                    if param.grad is not None and "token_embedding" in name:
                        param.grad.data *= gradient_mask.mean()

        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1

        return all_loss_dict

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        self.model.eval()
        total_metrics = {
            "conflict_precision": 0.0,
            "conflict_recall": 0.0,
            "conflict_f1": 0.0,
            "memory_accuracy": 0.0,
            "ppl": 0.0,
        }
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch.input_ids.to(self.device)
                attention_mask = batch.attention_mask.to(self.device)
                entity_ids = batch.entity_ids.to(self.device)
                conflict_positions = batch.conflict_positions.to(self.device)
                global_indices = batch.global_indices.to(self.device)

                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    entity_ids=entity_ids,
                    global_indices=global_indices,
                    detect_conflicts=True,
                    enable_rewrite=True,
                )

                pred_conflicts = outputs["conflict_logits"].argmax(dim=-1)
                true_conflicts = conflict_positions.long()

                tp = ((pred_conflicts == 1) & (true_conflicts == 1)).sum().float()
                fp = ((pred_conflicts == 1) & (true_conflicts == 0)).sum().float()
                fn = ((pred_conflicts == 0) & (true_conflicts == 1)).sum().float()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)

                if outputs.get("conflict_info") and outputs["conflict_info"].get("conflict_positions") is not None:
                    detected = outputs["conflict_info"]["conflict_positions"]
                    memory_accuracy = (detected == conflict_positions).float().mean()
                else:
                    memory_accuracy = torch.tensor(0.5)

                lm_logits = outputs["logits"]
                lm_loss = F.cross_entropy(
                    lm_logits.view(-1, self.config.vocab_size),
                    batch.labels.to(self.device).view(-1),
                    ignore_index=-100
                )
                ppl = torch.exp(lm_loss)

                total_metrics["conflict_precision"] += precision.item()
                total_metrics["conflict_recall"] += recall.item()
                total_metrics["conflict_f1"] += f1.item()
                total_metrics["memory_accuracy"] += memory_accuracy.item()
                total_metrics["ppl"] += ppl.item()
                num_batches += 1

        for k in total_metrics:
            total_metrics[k] /= num_batches

        return total_metrics


def create_training_pipeline(
    config: ModelConfig,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[Trainer, DataLoader, DataLoader]:
    model = MemoryAugmentedLLM(config)

    train_dataset = ContradictionDataset(config, num_samples=80, seed=42)
    val_dataset = ContradictionDataset(config, num_samples=20, seed=123)

    def collate_fn(batch):
        return ContradictionSample(
            input_ids=torch.stack([b.input_ids for b in batch]),
            attention_mask=torch.stack([b.attention_mask for b in batch]),
            entity_ids=torch.stack([b.entity_ids for b in batch]),
            labels=torch.stack([b.labels for b in batch]),
            conflict_positions=torch.stack([b.conflict_positions for b in batch]),
            conflict_entity_ids=torch.stack([b.conflict_entity_ids for b in batch]),
            correct_attribute_positions=torch.stack([b.correct_attribute_positions for b in batch]),
            global_indices=torch.stack([b.global_indices for b in batch]),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    trainer = Trainer(config, model, device)

    return trainer, train_loader, val_loader
