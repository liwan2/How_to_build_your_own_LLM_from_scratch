1. tokenizer.json —— 核心引擎（数据）
这是由 Hugging Face 自主研发的 tokenizers 库（Rust 后端）生成的核心文件。它包含了将文本切分成 Token 的所有具体算法和底层数据。

包含内容：

词汇表（Vocab）：所有 Token 到 ID 的完整映射字典。

合并规则（Merges）：BPE（字节对编码）、Unigram 等算法所需的规则表。

模型类型：明确指定分词器使用哪种底层算法（如 BPE、WordPiece 或 Unigram）。

后处理器（Post-processor）：自动添加 [CLS]/[SEP] 或 <s>/</s> 的逻辑。

特点：

体积较大（通常几 MB 到几十 MB）。

它是二进制的 Python 对象序列化，不可读性极强（全是乱码或嵌套极深的 JSON 数组）。

加载速度极快，因为它底层用 Rust 编写，处理文本的效率远高于纯 Python 实现。

2. tokenizer_config.json —— 配置参数（元数据）
这是一个轻量级的 JSON 格式配置文件，它决定了在调用 tokenizer() 方法时，默认的行为参数是什么。

包含内容：

特殊标记（Special Tokens）：定义 unk_token（未知词）、pad_token（填充）、bos_token（开始）、eos_token（结束）的具体字符串是什么。

默认处理参数：如 max_length、truncation（截断策略）、padding（填充策略）的默认值。

标准化规则：是否进行小写化（do_lower_case）、是否去除重音符号等。

类名映射：告诉 AutoTokenizer 该实例化哪个具体的 Python 类（例如 BertTokenizerFast）。

特点：

体积很小（几 KB）。

完全是纯文本 JSON，人类可直接阅读和修改。