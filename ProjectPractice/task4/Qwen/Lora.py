import torch
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from peft import LoraConfig, AutoPeftModelForCausalLM, TaskType, get_peft_model

model_path = r"D:\code\python\LearnPyLib\model\Generate\Qwen"
data_path = r"D:\code\python\LearnPyLib\model\Generate\data.json"
out_path = r"D:\code\python\LearnPyLib\model\Generate\Qwen\Qwen-LoRa-output"

def proc_func(example):
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
        tokenize=True, # 是否将instruction_text进行tokenize
        add_generation_prompt=False  # 如果设置了此条件，格式化输出后会附加一个带有标记的提示，表示助手消息的开头。
    )
    # 拿到问题input_ids
    prompt_ids = tokenizer.apply_chat_template(
        messages[0:-1], # 去掉最后的内容
        tokenize=True,
        add_generation_prompt=True
    )
    # 拿到问题长度
    prompt_length = len(prompt_ids)
    # 进行拼接
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    # 拿到掩码
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 导入数据
data = Dataset.from_pandas(pd.read_json(data_path))
# 创建Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# 处理数据
data_list = data.map(proc_func, remove_columns=data.column_names)
# 创建模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"cuda":0},
    torch_dtype=torch.bfloat16
)
model.enable_train_require_grads()
# 创建微调配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    r = 8,
    lora_alpha=32,
    lora_dropout=0.1
)
# 合并
model = get_peft_model(model, config)
# 创建训练参数
arg = TrainingArguments(
    output_dir=out_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=20,
    save_total_limit=2,
    bf16=True,
    learning_rate=1e-4,
    gradient_checkpointing=True,
)
# 开始训练
trainer = Trainer(
    model=model,
    args=arg,
    train_dataset=data_list,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=1024)
)
trainer.train()
# 保存适配器
trainer.save_model(out_path)
# 再次加载模型
model_ = AutoPeftModelForCausalLM.from_pretrained(
    out_path,
    device_map={"cuda":0},
    torch_dtype=torch.bfloat16
)
# 合并权重并保存
m = model_.merge_and_unload()
m.save_pretrained(out_path)
tokenizer.save_pretrained(out_path)