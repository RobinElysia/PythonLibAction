import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import pandas as pd
from peft import (
    PromptTuningConfig, get_peft_model, TaskType
)

model_path = r"/opt/model"
data_path = r"./data/data.json"
output_path = r"/opt/model/Qwen-PromptTuning-output"

def proc_func(example):
    """输出处理"""
    # 抽取用户输入
    instruction_text = example['instruction'] + example.get('input', '')
    # 拼接消息到整个message
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": instruction_text},
        {"role": "assistant", "content": example['output']}
    ]
    # 创建模板，拿到问题+答案的input_ids
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,  # 是否将instruction_text进行tokenize
        add_generation_prompt=False
        # 这部分为什么设置成False呢？因为我们有了{"role": "assistant", "content": example['output']}这个字段
        # 这是正确答案
    )
    # 拿到问题input_ids
    prompt_ids = tokenizer.apply_chat_template(
        messages[0:-1],  # 去掉最后的标准答案
        tokenize=True,
        add_generation_prompt=True
        # 但是这里，我们去掉了标准答案，我们需要模型自己生成
        # 生成好的内容与上述正确答案做loss计算，也就是下述的labels
    )
    # 拿到问题长度
    prompt_length = len(prompt_ids)
    # 进行拼接，码住问题，拿到生成的内容，我们需要做 loss
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    # 拿到掩码
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
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