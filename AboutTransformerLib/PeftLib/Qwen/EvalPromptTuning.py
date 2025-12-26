import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 路径
BASE_MODEL_PATH = r"/opt/model"
ADAPTER_PATH = r"/opt/model/Qwen-PromptTuning-output"

# Gemma
prompt_template = (
    "<start_of_turn>system\n"
    "你是一个AI助手<end_of_turn>\n"
    "<start_of_turn>user\n"
    "{instruction}{input}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map={"":0},
        local_files_only=True,
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    tokenize = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True,local_files_only=True)
    model.eval()
    return model, tokenize

def gen_resp(tokenizer, model, query, input_text=""):
    """
        生成响应
        传入：tokenizer分词器对象, model模型对象, RAG_prompt数据库检索+用户输入的str数据
        :return: response模型响应的str数据
        """
    # 构建消息列表
    messages = [
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": query}
    ]
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 字符串转张量
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # 生成
    with torch.no_grad():
        # generated_ids是一个二维的[1,n]的张量，n包含问题+答案，假设答案长度是 l
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )
    # 解码输出（跳过输入部分）
    # model_inputs.input_ids.shape[1]这是问题，截出后面是答案
    response_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    # 截的答案仍是二维[1,l]，需要转成1维
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model, tokenizer = load_model()
    query = "北京有什么好玩的么？"
    print(gen_resp(tokenizer, model, query))