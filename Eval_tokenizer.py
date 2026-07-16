from transformers import AutoTokenizer
import os

model_path="D:/AI_Project/Tokenizer/"
tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)

special_tokens = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<cot>",
    "</cot>",
    "<buffer1>",
    "<buffer2>",
    "<buffer3>",
    "<buffer4>",
    "<buffer5>",
    "<buffer6>",
    "<buffer7>",
    "<buffer8>",
    "<buffer9>",
    "<buffer10>",
    "<buffer11>",
    "<buffer12>",
    "<buffer13>",
]

tokenizer.add_special_tokens({
    "additional_special_tokens": special_tokens
})
#修复特殊token被拆散

messages = [

    {"role": "system", "content": "你是一个优秀的聊天机器人，请根据用户的提问，给出简洁、准确、专业的回答。"},
    {"role": "user", "content": '你是谁？'},
    {"role": "assistant", "content": '我是刘释阳'},
    {"role": "user", "content": '你到底是谁？'},
    {"role": "assistant", "content": '我是一个由OpenAI开发的人工智能聊天机器人，旨在与用户进行自然语言交流，提供信息和帮助。'},
    ]

new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
#模板化之后才可以进入分词器处理，生成模型输入#
print(new_prompt+'\n')
print('tokenizer词表长度：', len(tokenizer))
model_inputs = tokenizer(new_prompt)

print(model_inputs)
print('tokenizer编码后的长度：', len(model_inputs['input_ids']))

# for i in model_inputs['input_ids']:
#     print(i, tokenizer.decode(i, skip_special_tokens=False))
response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
print('一致性检查:', new_prompt == response)


test_texts = [
    # 中文 1：AI 与日常生活
    "人工智能正在深刻改变人们的日常生活，从智能语音助手到自动驾驶，从推荐算法到医疗辅助诊断，技术的落地速度远超以往。随着算力提升与数据积累，大模型不仅能理解语言、生成内容，还能辅助创作、规划任务甚至进行简单逻辑推理。未来人工智能将进一步与教育、金融、工业等领域深度融合，在提升效率的同时，也带来伦理、安全与就业结构等新的挑战。",

    # 中文 2：历史文化类
    "中国古代丝绸之路不仅是商贸通道，更是东西方文明交流的重要纽带。丝绸、茶叶、瓷器从东方运往西域，而葡萄、胡萝卜、佛教等也沿着这条道路传入中原。沿途的古城、石窟、驿站见证了无数商旅往来与文化碰撞，这种跨区域的交流促进了技术传播、艺术融合与民族交融，为后世多元文明共生提供了重要历史借鉴。",

    # 英文 1：Computer Science
    "Computer vision is a field of artificial intelligence that enables computers to understand meaningful information from digital images, videos and other visual inputs. It powers applications like facial recognition, autonomous driving, medical image analysis and industrial defect detection. Modern computer vision systems heavily rely on convolutional neural networks and transformer architectures to achieve high accuracy in classification, detection and segmentation tasks.",

    # 英文 2：Society & Education
    "Online learning has become an important part of modern education systems, offering flexible access to knowledge across geographical boundaries. It allows students to learn at their own pace, review materials repeatedly and access courses from top institutions worldwide. However, it also challenges traditional classroom models, requiring stronger self-discipline, better digital literacy and more reliable internet infrastructure to ensure effective learning outcomes.",

    # 中英混合：科技+生活双语
    "深度学习（Deep Learning）是机器学习的重要分支，近年来在自然语言处理领域取得巨大突破。Deep learning models can process massive amounts of text data and capture complex semantic relationships efficiently. 如今，聊天机器人、机器翻译、情感分析、文本摘要等工具已广泛应用于互联网产品，极大提升了信息处理效率。As AI continues to evolve, human-computer interaction will become more natural and intelligent in the near future."
]

ave=0
for i,text in enumerate(test_texts):
    char_sum=len(text)
    text_encoded=tokenizer.encode(text)
    token_sum=len(text_encoded)
    compress_ratio=char_sum/token_sum
    ave=(1/(i+1))*compress_ratio+(1-1/(i+1))*ave
    print(f'第{i}个句子：压缩率为{compress_ratio}')
print(f"平均压缩率：{ave}")

print('流式解码（每两个token作一组测试）：')
input_ids = model_inputs['input_ids']

i = 0
while i < len(input_ids):
    batch = input_ids[i:i+2]
    i += 2

    # 解码这一批
    decoded = tokenizer.decode(batch, skip_special_tokens=False)
    raw_tokens = tokenizer.convert_ids_to_tokens(batch)

    print(f'IDs: {str(batch):12} -> Tokens: {str(raw_tokens):25} -> Str: {repr(decoded)}')#repr，为结果加上引号，并显现原不显现的符号
