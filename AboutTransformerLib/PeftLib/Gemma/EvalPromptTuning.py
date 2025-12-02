import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 路径
BASE_MODEL_PATH = r"/opt/model"
ADAPTER_PATH = r"/opt/model/gemma2-PromptTuning-output"

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
    # 1. 格式化输入
    prompt = prompt_template.format(instruction=query, input=input_text)

    # 2. 编码
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 3. 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # 最大生成长度
            do_sample=True,  # 开启采样，增加多样性
            temperature=0.7,  # 温度控制
            top_p=0.9,  # 核采样
            repetition_penalty=1.1,  # 重复惩罚
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # 4. 解码 (只截取生成部分，去掉 Prompt)
    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response

if __name__ == "__main__":
    model, tokenizer = load_model()
    query = "北京有什么好玩的么？"
    print(gen_resp(tokenizer, model, query))