"""
预训练模型困惑度 (Perplexity) 评估脚本
在验证集上计算 loss 和 PPL，客观衡量预训练的语言建模能力
用法: python Eval_perplexity.py --data_path datasets/pretrain_t2t_mini.jsonl --max_samples 500
"""
import argparse
import torch
import math
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model.model_liwan import liwan_config, liwanForcasualLLM
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_model_params
warnings.filterwarnings('ignore')


def eval_ppl(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            res = model(input_ids, labels=labels)
            # loss 已经是 batch 内所有 token 的均值
            loss = res.loss.item()
            # 计算该 batch 中有效的 token 数 (labels != -100)
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(description="liwan 预训练模型困惑度评估")
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--weight', default='pretrain', type=str)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--max_seq_len', default=340, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--data_path', default='datasets/pretrain_t2t.jsonl', type=str, help="评估数据路径")
    parser.add_argument('--max_samples', default=1000, type=int, help="最多评估样本数（0=全量）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    args = parser.parse_args()

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained('./model')
    config = liwan_config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = liwanForcasualLLM(config)
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model = model.half()
    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.device).eval()
    get_model_params(model, config)

    # 加载评估数据
    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    if args.max_samples > 0 and args.max_samples < len(ds):
        ds.samples = ds.samples.select(range(args.max_samples))
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # 计算 PPL
    avg_loss, ppl = eval_ppl(model, loader, args.device)

    print(f"\n{'='*50}")
    print(f"  评估样本数: {len(ds)}")
    print(f"  Avg Loss:   {avg_loss:.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"{'='*50}")
    print(f"\n解读:")
    print(f"  PPL 越低越好。PPL = exp(loss)，表示模型平均在每个 token 上的'困惑程度'。")
    print(f"  参考: GPT-1 在 PTB 上 PPL≈70, LLaMA-7B 在 WikiText-2 上 PPL≈5-7")
    print(f"  (不同数据集和 tokenizer 的 PPL 不能直接比较，仅供参考)")


if __name__ == "__main__":
    main()
