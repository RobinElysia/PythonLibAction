import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import get_peft_model, TaskType, PrefixTuningConfig
import pandas as pd

model_path = r"/opt/model"
data_path = r"./data/data.json"
output_path = r"/opt/model/gemma2-PrefixTuning-output"

prompt_template = (
    "<start_of_turn>system\n"
    "你是一个AI助手<end_of_turn>\n"
    "<start_of_turn>user\n"
    "{instruction}{input}<end_of_turn>"
    "<start_of_turn>model\n"
)

# 处理函数
def proc_func(example):
    input = prompt_template.format(example["instruction"], example.get("input",""))
    response = f"{example['output']}"

    inp_ids = tokenizer(input, add_special_tokens=False)["input_ids"]
    res_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    input_ids = inp_ids + res_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(inp_ids) + res_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 加载数据
data = Dataset.from_pandas(pd.read_json(data_path))
# 初始化Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# 处理数据
data_handle = data.map(proc_func,  remove_columns=data.column_names)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"":0},
    torch_dtype=torch.bfloat16
)
model.enable_input_require_grads()
# 微调配置
config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=10,
    prefix_projection=True
    # 优势：
        # 可能提升模型性能
        # 允许更灵活的前缀表示学习
    # 注意事项：
        # 增加少量额外参数
        # 训练时间可能略微增加
)
# 合并模型+微调配置
model = get_peft_model(model, config)
# 配置微调参数
arg = TrainingArguments(
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
# 创建微调对象
trainer = Trainer(
    model=model,
    args=arg,
    train_dataset=data_handle,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=1024),
)
# 开始微调
trainer.train()
# 保存
trainer.save_model(output_path)
tokenizer.save_pretrained(output_path)