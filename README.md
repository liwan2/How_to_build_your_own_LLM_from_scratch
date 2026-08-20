# 从零实现 LLM

基于 PyTorch 从零实现的大型言模型（LLM）项目，覆盖从分词器训练、预训练、监督微调（SFT）、LoRA 微调到推理与评估的完整闭环。模型命名为 **liwan**。

本项目旨在以清晰、可读的代码展示 Decoder-only Transformer 的核心实现细节，同时具备工业级的工程能力（分布式训练、混合精度、断点续训、MoE 等）方便同我一样的学习者深入理解LLM的实现。

## 特性

- **完整模型实现**：Decoder-only Transformer，全部使用 PyTorch 原生模块从零搭建
- **现代架构组件**
  - RMSNorm 归一化
  - RoPE（旋转位置编码）+ YaRN 长度外推
  - GQA（分组查询注意力）
  - Flash Attention（`scaled_dot_product_attention`）
  - SwiGLU 前馈网络
  - MoE（混合专家）+ 专家负载均衡辅助损失
  - 词表嵌入与输出层权重绑定
- **自定义生成**：支持温度采样、top-k、top-p、重复惩罚、KV Cache、流式输出
- **BPE 分词器**：基于 HuggingFace `tokenizers` 训练，含完整 ChatML 对话模板与多模态预留 token
- **训练能力**：DDP 多卡分布式、混合精度（bfloat16/float16）、梯度累积、梯度裁剪、断点续训、学习率余弦退火
- **LoRA 微调**：基于 PEFT，支持训练后合并权重
- **评估**：预训练续写、困惑度（Perplexity）评估

## 目录结构

```
.
├── model/
│   ├── model_liwan.py         # 模型核心定义（config、Attention、MoE、FFN、生成）
│   ├── tokenizer.json         # 训练好的 BPE 分词器权重
│   └── tokenizer_config.json  # 分词器配置（含 ChatML chat_template）
├── Tokenizer/
│   ├── Tokenizer.py           # BPE 分词器训练脚本
│   └── Eval_tokenizer.py      # 分词器效果评估脚本
├── dataset/
│   └── lm_dataset.py          # 预训练数据集 & SFT 数据集
├── trainer/
│   ├── pretrain.py            # 预训练脚本
│   ├── train_full_sft.py      # 全量 SFT 微调脚本
│   ├── sft_lora.py            # LoRA 微调脚本
│   └── trainer_utils.py       # 训练工具函数（分布式、检查点、采样器等）
├── Eval_llm.py                # 预训练模型续写推理
├── Eval_perplexity.py         # 困惑度评估
└── .gitignore
```

## 模型架构

模型定义位于 `model/model_liwan.py`，整体结构如下：

```
liwanForcasualLLM (PreTrainedModel + GenerationMixin)
└── liwanModel
    ├── Embedding (vocab_size × hidden_size)
    ├── Dropout
    ├── liwanBlock × num_hidden_layer
    │   ├── input_layernorm (RMSNorm)
    │   ├── self_attn (Attention)
    │   │   ├── q_proj / k_proj / v_proj / o_proj
    │   │   ├── q_norm / k_norm (RMSNorm)
    │   │   └── RoPE + GQA + Flash Attention
    │   ├── post_attention_layernorm (RMSNorm)
    │   └── MLP (Feedback / MOEFeedback)
    └── norm (RMSNorm)
└── lm_head (Linear，与 Embedding 权重绑定)
```

### 核心组件说明

| 组件 | 实现 | 说明 |
| ---- | ---- | ---- |
| 归一化 | `RMSnorm` | 使用 RMSNorm，替代传统 LayerNorm |
| 位置编码 | `precompute_freqs_cis` / `apply_rotary_pos_emb` | 预计算 RoPE 正余弦表，支持 YaRN 外推 |
| 注意力 | `Attention` | GQA，q/k 各自 RMSNorm，KV Cache，Flash Attention |
| 前馈网络 | `Feedback` | SwiGLU：`down(act(gate(x)) * up(x))` |
| MoE 前馈 | `MOEFeedback` | 每个 token 选 top-k 专家，带负载均衡辅助损失 |
| 生成 | `generate` | 自回归解码 + 采样策略 + KV Cache |

### 默认配置（`liwan_config`）

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `hidden_size` | 768 | 隐藏层维度 |
| `num_hidden_layer` | 8 | Transformer 层数 |
| `num_attention_heads` | 8 | 注意力头数 |
| `num_key_value_heads` | 4 | KV 头数（GQA，n_rep=2） |
| `head_dim` | 96 | 每个头的维度 |
| `vocab_size` | 8000 | 词表大小 |
| `intermediate_size` | 768 向上取整到 64 的倍数 | FFN 中间层维度 |
| `hidden_act` | silu | 激活函数 |
| `max_position_embeddings` | 32768 | 最大位置编码长度 |
| `use_moe` | False | 是否启用 MoE |
| `num_experts` | 4 | MoE 专家数量 |
| `num_experts_per_tok` | 1 | 每个 token 激活的专家数 |
| `router_aux_loss_coef` | 5e-4 | 负载均衡辅助损失系数 |
| `tie_word_embeddings` | True | 输入嵌入与输出层权重绑定 |
| `flash_attn` | True | 是否使用 Flash Attention |
| `dropout` | 0.0 | dropout 概率 |

## 快速开始

### 环境依赖

- Python 3.8+
- [PyTorch](https://pytorch.org/)（建议 2.0+，以支持 `scaled_dot_product_attention`）
- [transformers](https://github.com/huggingface/transformers)
- [tokenizers](https://github.com/huggingface/tokenizers)
- [datasets](https://github.com/huggingface/datasets)（HuggingFace 的数据加载库）
- [peft](https://github.com/huggingface/peft)（仅 LoRA 微调需要）
- numpy、tqdm
- swanlab（可选，用于训练日志可视化）

安装示例：

```bash
pip install torch transformers tokenizers datasets peft numpy tqdm swanlab
```

### 数据格式

**预训练数据**（JSONL，每行一个 JSON 对象，包含 `text` 字段）：

```json
{"text": "这是一段用于预训练的文本。"}
```

**SFT 数据**（JSONL，每行一个 JSON 对象，包含 `conversations` 字段）：

```json
{
  "conversations": [
    {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么可以帮你的吗？"}
  ]
}
```

## 训练流程

推荐按以下顺序进行：

### 1. 训练分词器

在 `Tokenizer/Tokenizer.py` 顶部设置数据路径与保存路径，然后运行：

```bash
python Tokenizer/Tokenizer.py
```

脚本使用 ByteLevel BPE 算法训练分词器，并生成 `tokenizer.json` 与 `tokenizer_config.json`。后者包含 ChatML 对话模板、多模态预留 token（vision/audio/video）以及工具调用（tool_call）相关 token。

### 2. 预训练

```bash
python trainer/pretrain.py \
    --data_path <预训练数据路径> \
    --save_dir <模型保存目录> \
    --epochs 2 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --max_seq_len 340 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --use_moe 0
```

关键参数：

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `--data_path` | — | 预训练数据路径 |
| `--save_dir` | — | 模型保存目录 |
| `--save_weight` | `pretrain` | 保存权重的前缀名 |
| `--epochs` | 2 | 训练轮数 |
| `--batch_size` | 32 | batch size |
| `--learning_rate` | 5e-4 | 学习率 |
| `--accumulation_steps` | 8 | 梯度累积步数 |
| `--max_seq_len` | 340 | 最大序列长度 |
| `--use_moe` | 0 | 是否使用 MoE（0/1） |
| `--dtype` | bfloat16 | 混合精度类型 |
| `--from_weight` | none | 基于哪个权重继续训练 |
| `--from_resume` | 0 | 是否自动检测并续训 |
| `--use_compile` | 0 | 是否使用 torch.compile 加速 |
| `--use_wandb` | — | 是否启用 swanlab 日志 |

### 3. 全量 SFT 微调

```bash
python trainer/train_full_sft.py \
    --data_path <SFT 数据路径> \
    --save_dir <模型保存目录> \
    --from_weight pretrain \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --max_seq_len 768
```

SFT 训练中会以 `--add_system_ratio`（默认 0.2）的概率随机注入 system prompt，并按角色对 `assistant` 回复部分计算损失（其余部分 label 置为 -100）。

### 4. LoRA 微调

```bash
python trainer/sft_lora.py \
    --data_path <LoRA 数据路径> \
    --base_weight <基础模型权重名> \
    --lora_save_dir <LoRA 保存目录> \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --lora_r 8 \
    --lora_alpha 16
```

LoRA 通过 PEFT 自动发现模型中所有线性层作为目标模块，训练后可加 `--merge` 参数将 LoRA 权重合并回基础模型并保存完整权重。还可通过 `--mix_data` 混入通用 SFT 数据以缓解灾难性遗忘。

## 推理与评估

### 预训练模型续写

```bash
python Eval_llm.py \
    --weight <权重名> \
    --save_dir <模型保存目录> \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --max_new_tokens 512
```

运行后可选 `[0] 自动测试` 或 `[1] 手动输入`。预训练模型只会续写，不具备对话能力。

### 困惑度评估

```bash
python Eval_perplexity.py \
    --weight <权重名> \
    --data_path <评估数据路径> \
    --max_seq_len 340 \
    --batch_size 16 \
    --max_samples 1000
```

脚本在验证集上计算平均 Loss 与 Perplexity（PPL = exp(loss)），用于客观衡量预训练语言建模能力。PPL 越低越好。

### 分词器评估

```bash
python Tokenizer/Eval_tokenizer.py
```

脚本用于验证分词器的特殊 token 完整性、chat_template 渲染、编解码一致性，以及中英文文本的压缩率。

## 训练工具说明

`trainer/trainer_utils.py` 提供了以下通用能力：

- `init_distributed_mode`：初始化 DDP 分布式训练（NCCL 后端）
- `setup_seed`：固定随机种子，保证可复现
- `get_lr`：余弦退火学习率调度
- `lm_checkpoint`：模型保存与断点续训（含 optimizer、scaler、wandb 状态）
- `init_model`：初始化分词器与模型，并按权重前缀加载
- `SkipBatchSampler`：按批次采样并可跳过前 N 个 batch（用于续训）
- `get_model_params`：统计并打印模型总参数与激活参数（MoE 场景）

## 常见问题

- **中文 1 token 约等于 1.5~1.7 个字符**，设置 `max_seq_len` 时需考虑这一点。
- 训练使用 bfloat16 时无需 GradScaler，float16 时自动启用。
- MoE 模式下会额外输出 `aux_loss`（专家负载均衡损失），总损失 = 交叉熵损失 + aux_loss。
- 若使用多卡训练，需通过 `torchrun` 或相应启动器启动，并设置 `RANK`、`LOCAL_RANK` 等环境变量。

## 说明

本项目为学习/教学用途的从零实现，代码结构清晰、注释详尽，适合用于理解 LLM 的核心原理与训练流程。架构设计参考了 LLaMA 系列等主流开源模型的公开设计思想。
