import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = r"/opt/model/merged_model"

# Llama模板语法
PROMPT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "你是一个AI助手。<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}{input}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


def load_model():
    """
    加载模型
    :return: tokenizer分词器对象, model模型对象
    """
    print("🔄 正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 这一步与训练时保持一致，加载基座
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"cuda":0},
        torch_dtype=torch.bfloat16,  # 保持和训练时一致的精度
        trust_remote_code=True
    )
    model.eval()  # 切换到评估模式
    return tokenizer, model


def gen_resp(tokenizer, model, query, input_text=""):
    """
    生成响应
    传入：tokenizer分词器对象, model模型对象, RAG_prompt数据库检索+用户输入的str数据
    :return: response模型响应的str数据
    """
    # 1. 格式化输入
    prompt = PROMPT_TEMPLATE.format(instruction=query, input=input_text)

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
    # 加载模型
    tokenizer, model = load_model()

    # 待验证的虚构旅游地列表
    test_queries = [
        "介绍一下北京地摊文化博物馆。",
        "请问新疆海洋文化博物馆在哪里？有什么特色？",
        "给我讲讲北京地摊文化博物馆的镇馆之宝是什么？",
        "我想去新疆海洋文化博物馆旅游，有什么建议吗？"
    ]

    print("🚀 开始验证模型对虚构数据的掌握能力")

    for query in test_queries:
        print(f"👤 User: {query}")
        response = gen_resp(tokenizer, model, query)
        print(f"🤖 Assistant: {response}")
        print("-" * 50)