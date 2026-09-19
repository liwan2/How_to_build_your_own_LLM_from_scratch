"""
LoRA 微调脚本 —— Qwen 版
用法:
    # 训练数据集
    python trainer/sft_lora_qwen.py --data_path My_secret_datasets/minister_data.jsonl

    # 指定模型路径和输出目录
    python trainer/sft_lora_qwen.py \
        --data_path My_secret_datasets/minister_data.jsonl \
        --model_path D:/AI_Project/qwen_model \
        --lora_save_dir out/lora_minister
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from contextlib import nullcontext

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings('ignore')


# ==================== 数据集 ====================

class QwenSFTDataset(Dataset):
    """Qwen 格式的 SFT 数据集，处理 ChatML 格式对话"""
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                import json
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["conversations"]

        # 用 Qwen 自带的 chat template 格式化
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # 编码
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )

        input_ids = encodings["input_ids"]
        labels = input_ids.copy()

        # 把 prompt 部分的 label 设为 -100，只训练 assistant 的回复
        # 找到最后一个 <|im_start|>assistant 的位置
        # 简化处理：所有 user/system 的内容都不计算 loss，只算 assistant 的
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # 找到所有 assistant 开始的位置
        trainable_start = None
        for i in range(len(input_ids) - 1):
            if input_ids[i] == im_start_id and input_ids[i+1] == assistant_id:
                trainable_start = i + 2  # 跳过 <|im_start|> 和 "assistant"
                # 把之前的都设为 -100
                for j in range(i):
                    labels[j] = -100
                break  # 只处理第一轮，简化

        # 如果没找到 assistant，全部设为 -100（不会发生）
        if trainable_start is None:
            labels = [-100] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def collate_fn(batch):
    """padding 到 batch 内最大长度"""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []

    pad_id = 151643  # Qwen 的 pad_token_id

    for x in batch:
        ids = x["input_ids"]
        lab = x["labels"]
        pad_len = max_len - len(ids)

        input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask)
    }


# ==================== LoRA 相关 ====================

def get_target_modules(model):
    """自动发现模型中的所有线性层名称"""
    targets = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layer_name = name.split('.')[-1]
            if layer_name not in ("lm_head", "embed_tokens"):
                targets.add(layer_name)
    return sorted(targets)


def apply_lora(model, r=8, alpha=16, dropout=0.05):
    """给模型挂 LoRA 适配器"""
    target_modules = get_target_modules(model)
    print(f"[INFO] LoRA 目标模块: {target_modules}")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    return model


# ==================== 训练循环 ====================

def train_epoch(epoch, loader, iters, start_step=0):
    start_time = time.time()
    last_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        last_step = step

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = res.loss
            loss = loss / args.accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % args.accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend = time.time() - start_time
            eta = spend / max(step - start_step, 1) * (iters - step) // 60
            print(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {loss.item() * args.accumulation_steps:.4f}, "
                f"lr: {optimizer.param_groups[-1]['lr']:.8f}, "
                f"eta: {eta:.0f}min"
            )

        if step % args.save_interval == 0 or step == iters:
            save_checkpoint(epoch, step)

        del input_ids, labels, attention_mask, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def get_lr(step, total_steps, base_lr):
    """余弦学习率衰减"""
    import math
    warmup_steps = int(0.1 * total_steps)
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(epoch, step):
    """保存 LoRA 权重"""
    os.makedirs(args.lora_save_dir, exist_ok=True)
    lora_path = os.path.join(args.lora_save_dir, "adapter_model")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"[INFO] LoRA 权重已保存到: {lora_path}  (epoch={epoch+1}, step={step})")


# ==================== 主函数 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen LoRA 微调")
    # 数据 & 模型
    parser.add_argument("--data_path", type=str, default="My_secret_datasets/minister_data.jsonl")
    parser.add_argument("--model_path", type=str, default="D:/AI_Project/qwen_model",
                        help="Qwen 模型路径")
    parser.add_argument("--lora_save_dir", type=str, default="out/lora_minister",
                        help="LoRA 权重保存目录")
    parser.add_argument("--merge", action="store_true",
                        help="训练结束后合并 LoRA 并保存完整模型")

    # 训练超参
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=1024)

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # 设备
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    # ========== 1. 初始化 ==========
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ========== 2. 混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if device_type == "cpu":
        autocast_ctx = nullcontext()
        scaler = None
    elif args.dtype == "float16":
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
        scaler = torch.cuda.amp.GradScaler()
    else:
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
        scaler = None

    # ========== 3. 加载模型和 tokenizer ==========
    print(f"[INFO] 加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True
    )
    print(f"[INFO] 模型加载完成")

    # ========== 4. 挂 LoRA ==========
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                       dropout=args.lora_dropout)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] LoRA 可训练参数: {trainable:,} / {total:,}  ({100 * trainable / total:.2f}%)")

    # ========== 5. 加载数据 ==========
    data_path = os.path.join(os.path.dirname(__file__), '..', args.data_path)
    data_path = os.path.normpath(data_path)
    print(f"[INFO] 数据路径: {data_path}")
    train_ds = QwenSFTDataset(data_path, tokenizer, max_length=args.max_seq_len)
    print(f"[INFO] 数据集: {len(train_ds)} 条")

    # ========== 6. 优化器 ==========
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # ========== 7. DataLoader ==========
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
        collate_fn=collate_fn
    )
    iters = len(loader)

    # ========== 8. 训练 ==========
    print(f"\n{'='*50}")
    print(f"开始 LoRA 训练 (Qwen 版)")
    print(f"  数据: {len(train_ds)} 条 | Batch: {args.batch_size} | "
          f"累积: {args.accumulation_steps} | Epochs: {args.epochs}")
    print(f"  LoRA r={args.lora_r} alpha={args.lora_alpha} "
          f"dropout={args.lora_dropout}")
    print(f"  LR: {args.learning_rate} | 有效 BS: "
          f"{args.batch_size * args.accumulation_steps}")
    print(f"{'='*50}\n")

    for epoch in range(args.epochs):
        torch.manual_seed(42 + epoch)
        train_epoch(epoch, loader, iters)

    # 最终保存
    save_checkpoint(args.epochs - 1, iters)

    # ========== 9. 合并 LoRA → 完整模型 ==========
    if args.merge:
        print("\n[INFO] 合并 LoRA 到基础模型...")
        merged = model.merge_and_unload()
        merge_path = args.lora_save_dir + "_merged"
        merged.save_pretrained(merge_path)
        tokenizer.save_pretrained(merge_path)
        print(f"[INFO] 合并模型已保存: {merge_path}")

    print("\n训练完成!")
