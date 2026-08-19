"""
LoRA 微调脚本 —— 女友风格注入
用法:
    # 训练
    python trainer/sft_lora.py --data_path My_secret_datasets/girlfriend_data.jsonl

    # 训练后合并 LoRA + 保存完整模型
    python trainer/sft_lora.py --data_path My_secret_datasets/girlfriend_data.jsonl --merge

    # 指定基础模型
    python trainer/sft_lora.py --data_path My_secret_datasets/girlfriend_data.jsonl \
        --base_weight full_sft_v2 --hidden_size 768 --num_hidden_layers 8
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
from torch.utils.data import DataLoader
from contextlib import nullcontext

from model.model_liwan import liwan_config, liwanForcasualLLM
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr, Logger, init_distributed_mode, setup_seed,
    get_model_params, SkipBatchSampler
)

warnings.filterwarnings('ignore')

# ==================== LoRA 相关 ====================

def get_target_modules(model):
    """自动发现模型中的所有线性层名称（排除 lm_head/embed_tokens）"""
    targets = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layer_name = name.split('.')[-1]
            # lm_head 与 embedding 权重绑定，不能挂 LoRA
            if layer_name not in ("lm_head", "embed_tokens"):
                targets.add(layer_name)
    return sorted(targets)


def apply_lora(model, r=8, alpha=16, dropout=0.05):
    """给模型挂 LoRA 适配器"""
    target_modules = get_target_modules(model)
    Logger(f"LoRA 目标模块: {target_modules}")

    from peft import LoraConfig, get_peft_model, TaskType

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

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss
            if hasattr(res, 'aux_loss') and res.aux_loss is not None:
                loss = loss + res.aux_loss
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
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {loss.item() * args.accumulation_steps:.4f}, "
                f"lr: {optimizer.param_groups[-1]['lr']:.8f}, "
                f"eta: {eta:.0f}min"
            )

        if step % args.save_interval == 0 or step == iters:
            save_checkpoint(epoch, step)

        del input_ids, labels, res, loss

    # 尾部残余梯度处理
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


def save_checkpoint(epoch, step):
    """保存 LoRA 权重"""
    os.makedirs(args.lora_save_dir, exist_ok=True)
    lora_path = os.path.join(args.lora_save_dir, "adapter_model")
    model.save_pretrained(lora_path)
    Logger(f"LoRA 权重已保存到: {lora_path}  (epoch={epoch+1}, step={step})")


# ==================== 主函数 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="liwan LoRA 微调")
    # 数据 & 模型
    parser.add_argument("--data_path", type=str, default="My_secret_datasets/girlfriend_data.jsonl")
    parser.add_argument("--base_weight", type=str, default="full_sft_v2",
                        help="基础模型权重名 (从 out/ 目录加载)")
    parser.add_argument("--lora_save_dir", type=str, default="out/lora_girlfriend",
                        help="LoRA 权重保存目录")
    parser.add_argument("--merge", action="store_true",
                        help="训练结束后合并 LoRA 并保存完整模型")
    parser.add_argument("--mix_data", type=str, default="",
                        help="混入通用 SFT 数据防止遗忘，如 My_secret_datasets/sft_final.jsonl")

    # 训练超参
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=200)

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # 模型结构 (需与基础权重匹配)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])

    # 设备
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()
    args.use_moe = bool(args.use_moe)

    # ========== 1. 初始化 ==========
    local_rank = init_distributed_mode()
    if torch.distributed.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42)

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
        scaler = None  # bfloat16 不需要 scaler

    # ========== 3. 加载基础模型 ==========
    Logger(f"加载基础模型: out/{args.base_weight}_{args.hidden_size}.pth")
    lm_config = liwan_config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
    )
    model = liwanForcasualLLM(lm_config)

    ckp_path = f"out/{args.base_weight}_{args.hidden_size}.pth"
    state_dict = torch.load(ckp_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    del state_dict

    model = model.half().to(args.device)
    get_model_params(model, lm_config)

    # ========== 4. 挂 LoRA ==========
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                       dropout=args.lora_dropout)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    Logger(f"LoRA 可训练参数: {trainable:,} / {total:,}  ({100 * trainable / total:.2f}%)")

    # ========== 5. 加载 tokenizer & 数据 ==========
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('./model')
    tokenizer.model_max_length = args.max_seq_len

    data_path = os.path.join(os.path.dirname(__file__), '..', args.data_path)
    data_path = os.path.normpath(data_path)
    Logger(f"数据路径: {data_path}")
    train_ds = SFTDataset(data_path, tokenizer, max_length=args.max_seq_len)
    Logger(f"女友数据集: {len(train_ds)} 条")

    # 混入通用 SFT 数据防止灾难性遗忘
    if args.mix_data:
        mix_path = os.path.join(os.path.dirname(__file__), '..', args.mix_data)
        mix_path = os.path.normpath(mix_path)
        Logger(f"混入通用数据: {mix_path}")
        mix_ds = SFTDataset(mix_path, tokenizer, max_length=args.max_seq_len)
        # 抽样通用数据，控制在女友数据的 2~3 倍
        sample_size = min(len(mix_ds), len(train_ds) * 3)
        indices = torch.randperm(len(mix_ds))[:sample_size].tolist()
        mix_ds = torch.utils.data.Subset(mix_ds, indices)
        Logger(f"通用数据采样: {len(mix_ds)} 条 (总量 {sample_size})")
        train_ds = torch.utils.data.ConcatDataset([train_ds, mix_ds])
        Logger(f"合并后数据集: {len(train_ds)} 条")

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
    )
    iters = len(loader)

    # ========== 8. 训练 ==========
    Logger(f"\n{'='*50}")
    Logger(f"开始 LoRA 训练")
    Logger(f"  数据: {len(train_ds)} 条 | Batch: {args.batch_size} | "
           f"累积: {args.accumulation_steps} | Epochs: {args.epochs}")
    Logger(f"  LoRA r={args.lora_r} alpha={args.lora_alpha} "
           f"dropout={args.lora_dropout}")
    Logger(f"  LR: {args.learning_rate} | 有效 BS: "
           f"{args.batch_size * args.accumulation_steps}")
    Logger(f"{'='*50}\n")

    for epoch in range(args.epochs):
        setup_seed(42 + epoch)
        train_epoch(epoch, loader, iters)

    # 最终保存
    save_checkpoint(args.epochs - 1, iters)

    # ========== 9. 合并 LoRA → 完整模型 ==========
    if args.merge:
        Logger("\n合并 LoRA 到基础模型...")
        merged = model.merge_and_unload()
        merged = merged.half()
        merge_path = f"out/{args.base_weight}_girlfriend_{args.hidden_size}.pth"
        torch.save(merged.state_dict(), merge_path)
        Logger(f"完整模型已保存: {merge_path}")
        Logger(f"推理命令: python test_sft.py --weight {args.base_weight}_girlfriend")

    Logger("\n训练完成!")
