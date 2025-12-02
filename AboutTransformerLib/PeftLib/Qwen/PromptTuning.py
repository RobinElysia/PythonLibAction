import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import pandas as pd
from peft import (
    PromptTuningConfig, get_peft_model, TaskType
)

model_path = r"/opt/model"
data_path = r"./data/data.json"
output_path = r"/opt/model/gemma2-PromptTuning-output"

# Gemma模板语法
prompt_template = (
    "<start_of_turn>system\n"
    "你是一个AI助手<end_of_turn>\n"
    "<start_of_turn>user\n"
    "{instruction}{input}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

def proc_func(example):
    """输出处理"""
    t = prompt_template.format(instruction=example["instruction"], input=example.get("input", ""))
    response = f"{example['output']}<end_of_turn>"

    inp_ids = tokenizer(t, add_special_tokens=False)["input_ids"]
    res_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    input_ids = inp_ids + res_ids
    attention_mask = [1] * len(input_ids)
    lab = [-100] * len(inp_ids) + res_ids

    # 判断过大值
    if len(input_ids) > 2048:
        input_ids = input_ids[:1024]
        attention_mask = attention_mask[:1024]
        lab = lab[:1024]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": lab
    }

# 加载数据
data = Dataset.from_pandas(pd.read_json(data_path))
# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# 调用map函数进行处理数据
data_to = data.map(proc_func, remove_columns=data.column_names)
# 创建模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"":0},
    torch_dtype=torch.bfloat16
)
model.enable_input_require_grades()
# 创建微调配置
confi = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init_text="你是一个旅游助手",
    num_virtual_tokens=len(tokenizer("你是一个旅游助手")["input_ids"]),
    tokenizer_name_or_path=tokenizer
)
# 微调模型封装
model = get_peft_model(model, confi)
# 创建训练参数
args = TrainingArguments(
    output_dir = output_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    logging_steps=20,
    save_total_limit=3,
    learning_rate = 1e-4,
    bf16=True,
    num_train_epochs=10,
)
# 创建训练器
trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=data_to,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=1024),
)
# 训练
trainer.train()
# 保存适配器
# 这会在 output_path 下生成 adapter_model.safetensors 和 adapter_config.json
trainer.save_model(output_path)
tokenizer.save_pretrained(output_path)
# Prompt不支持基座与微调合并，你需要再使用时进行再加载