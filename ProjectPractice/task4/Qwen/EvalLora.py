import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = r"D:\code\python\LearnPyLib\model\Generate\Qwen\Qwen-LoRa-output"

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"cuda":0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    return model, tokenizer


def gen_resp(model, tokenizer, query, input=""):
    # 构建消息列表
    messages = [
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": query}
    ]
    if input:
        messages.append({"role": "assistant", "content": input})
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 模型生成
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    # 解码输出（跳过输入部分）
    response_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model, tokenizer = load_model()
    query = "北京有什么好玩的？"
    print(gen_resp(model, tokenizer, query))