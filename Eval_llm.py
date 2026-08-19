import time
import argparse
import random
import torch
import warnings
from transformers import AutoTokenizer, TextStreamer
from model.model_liwan import liwan_config, liwanForcasualLLM
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_liwan_model(args):
    tokenizer = AutoTokenizer.from_pretrained('./model')
    config = liwan_config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = liwanForcasualLLM(config)
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location='cpu')
    # 先转 half 再 load，确保参数 dtype 与 checkpoint 一致
    model = model.half()
    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.device).eval()
    get_model_params(model, config)
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="liwan 预训练模型推理")
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--weight', default='pretrain', type=str)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--max_new_tokens', default=512, type=int)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    args = parser.parse_args()

    # 预训练模型测试prompt — 预训练模型只会续写，不会对话
    prompts = [
        '人工智能的发展经历了',
        '中国的首都是',
        'Python是一种',
        '光合作用是指',
        '机器学习是人工智能的一个分支，',
        '今天天气很好，',
        '在深度学习中，反向传播算法',
        '地球绕太阳公转一周需要',
    ]

    model, tokenizer = init_liwan_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('📝: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0:
            print(f'📝: {prompt}')
        # 预训练模型：直接用 bos_token + prompt 续写
        inputs = tokenizer(tokenizer.bos_token + prompt, return_tensors="pt", truncation=True).to(args.device)

        print('🤖: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.5
        )
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n')

if __name__ == "__main__":
    main()
