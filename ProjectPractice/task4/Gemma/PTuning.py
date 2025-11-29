import torch
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import TaskType, get_peft_model, PromptEncoderConfig

model_path = r"/opt/model"
data_path = r"./data/data.json"
output_path = r"/opt/model/gemma2-PTuning-output"

# 模板语法
prompt_template = (
    "<start_of_turn>system\n"
    "你是一个AI助手<end_of_turn>\n"
    "<start_of_turn>user\n"
    "{instruction}{input}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

def proc_func(example):
    input = prompt_template.format(example["instruction"], example.get("input", ""))
    response = f"{example['output']}<end_of_turn>"

    ins_ids = tokenizer(input, add_specilal_tokens=False)["input_ids"]
    res_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    input_ids = ins_ids + res_ids
    attention_mask = [1] * input_ids
    label = [-100] * len(ins_ids) + res_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": label
    }

# 加载数据
data = Dataset.from_pandas(pd.read_json(data_path))
# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# 数据处理
data_handel = data.map(proc_func, remove_columns=data.column_names)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"cuda":0},
    torch_dtype=torch.bfloat16,
)
model.enable_input_require_grads()
# 加载微调配置（使用MLP默认是v2）
config = PromptEncoderConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=10,
    encoder_dropout=0.1,
    encoder_num_layers=5,
    encoder_hidden_size=1024
)
# 合并模型
model = get_peft_model(model, config)
# 训练配置
args = TrainingArguments(
    output_dir = output_path,
    per_device_train_batch_size=1,
    logging_steps=20,
    save_steps=20,
    save_total_limit=2,
    num_train_epochs=3,
    learning_rate=1e-4,
    bf16=True,
    gradient_checkpointing=True,
)
# 训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=data_handel,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=1024),
)
# 保存
trainer.save_model(output_path)
tokenizer.save_pretrained()