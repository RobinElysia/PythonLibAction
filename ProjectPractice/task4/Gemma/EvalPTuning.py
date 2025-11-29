import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 路径
BASE_MODEL_PATH = r"/opt/model"
ADAPTER_PATH = r"/opt/model/gemma2-PTuning-output"

# 提示词模板
prompt_template = (
    "<start_of_turn>system\n"
    "你是一个AI助手<end_of_turn>\n"
    "<start_of_turn>user\n"
    "{instruction}{input}<end_of_turn>\n"
    "<start_of_turn>model\n"
)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map={"cuda":0},
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    return model, tokenizer

def gen_resp(tokenizer, model, query, input=""):
    prompt = prompt_template.format(instruction=query, input=input)
    input_ids = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output = model.generate(
            **input_ids,
            max_new_tokens=1024,
            do_sample=True
        )
    input_len = input_ids.input_ids.shape[1]
    gener_ids = output[0][input_len:]
    response = tokenizer.decode(gener_ids, skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model, tokenizer = load_model()
    query = "北京有什么好玩的么？"
    print(gen_resp(tokenizer, model, query))