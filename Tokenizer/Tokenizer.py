import os
import json
from tokenizers import decoders,models,pre_tokenizers,trainers,Tokenizer

Dataset_path='datasets/sft_t2t_mini.jsonl'
Tokenizer_path='Tokenizer/'
Vocabulary_size=8000
Special_tokens_num=40

#读入数据与处理数据
def get_texts(data_path):
    with open(data_path,'r',encoding='utf-8',errors='ignore') as f:
        for i,line in enumerate(f):
            try:
                data=json.loads(line.strip())
                contents=[]
                content=data.get('conversations',[])
                for item in content:
                    item2=item.get('content')
                    if item2:
                        contents.append(item2)
                if contents:
                    yield '\n'.join(contents)
            except json.JSONDecodeError:
                continue

#初始化分词器和预分词器
def train_tokenizer(dataset_path,tokenizer_path,vocab_size,special_tokens_num):
    tokenizer=Tokenizer(models.BPE(continuing_subword_prefix="",end_of_word_suffix=""))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)

    Special_tokens=[
        # 基础对话格式（ChatML）
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",  
        # 多模态相关Token
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"
    ]

    additional_tokens=[
        "<tool_call>", "</tool_call>",   
        "<tool_response>", "</tool_response>",  
        "<cot>", "</cot>"   
    ]
    #生成纯净的占位符Token，方便后续拓展
    num_buffer=Special_tokens_num-len(additional_tokens)-len(Special_tokens)
    buffer_tokens=[f"<buffer{i}>" for i in range(1,num_buffer+1)]
    total_special_tokens=Special_tokens+additional_tokens+buffer_tokens

    #初始化BPE训练器
    trainer=trainers.BpeTrainer(
        vocab_size=Vocabulary_size,
        special_tokens=total_special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    #tokenizer.add_special_tokens(total_special_tokens),这句话添加的特殊token是在训练之后，有可能会识别不到

    #解码器设置
    tokenizer.decoder=decoders.ByteLevel(add_prefix_space=False)

    #训练执行
    texts=get_texts(Dataset_path)
    tokenizer.train_from_iterator(texts,trainer=trainer)

    tokenizer.add_special_tokens(total_special_tokens)#冗余保险

    #模型保存
    os.makedirs(Tokenizer_path, exist_ok=True)
    tokenizer.save(os.path.join(Tokenizer_path,'tokenizer.json'))

    #transformers兼容配置文件
    added_tokens_decoder={}
    with open(os.path.join(Tokenizer_path,'tokenizer_config.json'),'w',encoding='utf-8') as f:
        for i,token in enumerate(total_special_tokens):
            idx=tokenizer.token_to_id(token)
            added_tokens_decoder[str(idx)]={
                "content":token,
                "single_word":True,
                "lstrip":False,
                "rstrip":False,
                "normalized":False,
                "special": True if (token in Special_tokens or token.startswith("<buffer")) else False
            }
        config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False, 
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [t for t in Special_tokens if t not in ["<|endoftext|>"]],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,  # 使用旧版ByteLevel解码逻辑，避免流式解码乱码
        "model_max_length": 131072,  # 模型支持的最大上下文长度
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        # 多模态Token配置
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        # 对话模板（Jinja2格式）
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}", 
        "tokenizer_class": "PreTrainedTokenizerFast"
        }
        json.dump(config,f,ensure_ascii=False,indent=4)

    print("Tokenizer training completed!")


if  __name__ == '__main__':
    train_tokenizer(Dataset_path,Tokenizer_path,Vocabulary_size,Special_tokens_num)







