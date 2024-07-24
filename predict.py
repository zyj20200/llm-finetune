import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def predict(messages, model, tokenizer):
    device = "cuda"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained("/mnt/models/Qwen2-72B-Instruct", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/mnt/models/Qwen2-72B-Instruct", device_map="auto", torch_dtype=torch.bfloat16)

# 加载训练好的Lora模型，将下面的[checkpoint-XXX]替换为实际的checkpoint文件名名称
model = PeftModel.from_pretrained(model, model_id="/mnt/train/Qwen2/examples/sft/output_qwen/checkpoint-35")


messages_list = [
    [{"role": "system", "content": "你是一位小学数学专家，能够帮助学生解决数学难题，提高数学能力，轻松掌握数学知识。"}, {"role": "user", "content": "数学题目：把30kg糖分成质量相等的多少袋，每袋的质量(10/3)千克．"}],
    [{"role": "system", "content": "你是一位小学数学专家，能够帮助学生解决数学难题，提高数学能力，轻松掌握数学知识。"}, {"role": "user", "content": "数学题目：有6个棱长分别是4厘米、5厘米、6厘米的相同的长方体，把它们的某些面染上红色，使得6个长方体中染有红色的面恰好分别是1个面、2个面、3个面、4个面、5个面和6个面．染色后把所有长方体分割成棱长为1厘米的小正方体，分割完毕后，恰有一面是红色的小正方体最多有多少个？"}],
    [{"role": "system", "content": "你是一位小学数学专家，能够帮助学生解决数学难题，提高数学能力，轻松掌握数学知识。"}, {"role": "user", "content": "数学题目：长方体的体积是36立方米，长是6米，宽是3米，高是多少米．"}]
]

for messages in messages_list:
    response = predict(messages, model, tokenizer)
    print(response)