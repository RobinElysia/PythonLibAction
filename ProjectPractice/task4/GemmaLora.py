import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq,
    TrainingArguments, Trainer
)
from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM

# --- 1. 配置常量 ---
MODEL_PATH = r"D:\code\python\LearnPyLib\model\Generate\Gemma"
# 训练数据路径（JSON格式指令集）
DATA_PATH = r"D:\code\python\LearnPyLib\model\Generate\Gemma\data.json"
# 输出路径
OUTPUT_DIR = r"D:\code\python\LearnPyLib\model\Generate\Gemma"

PROMPT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "你是一个有用的助手。<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}{input}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


# --- 2. 数据处理函数 ---
def process_func(example):
    instruction = PROMPT_TEMPLATE.format(instruction=example['instruction'], input=example.get('input', ''))
    response = f"{example['output']}<|eot_id|>"

    # add_special_tokens=False，因为你的模板里已经手动加了 <|begin_of_text|> 等
    ins_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    res_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    # 此时 input_ids 的长度是变长的
    input_ids = ins_ids + res_ids

    # 指令部分 (ins_ids) 设为 -100，表示计算 Loss 时忽略
    # 回答部分 (res_ids) 设为原 ID，表示需要学习
    labels = [-100] * len(ins_ids) + res_ids

    # 因为这里没有 Pad，所以全是有效内容，全是 1
    attention_mask = [1] * len(input_ids)

    # 在 process_func 返回前加一行防御性截断（只切不补）
    if len(input_ids) > 2048:  # 或者 2048，根据你显存决定
        input_ids = input_ids[:2048]
        attention_mask = attention_mask[:2048]
        labels = labels[:2048]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# --- 3. 主流程 ---
if __name__ == "__main__":
    # 3.1 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 3.2 准备数据
    ds = Dataset.from_pandas(pd.read_json(DATA_PATH))
    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
    # 这是一个Dataset类型数据，里面包含input_ids、attention_mask、labels三个字段
        # attention_mask:告诉模型哪些 token 是有效的
        # labels:训练时的“正确答案”

    # 3.3 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"cuda":0},
        torch_dtype=torch.bfloat16  # 推荐 bfloat16
    )
    model.enable_input_require_grads()  # 开启梯度检查点兼容

    # 3.4 配置 LoRA (自动冻结非LoRA参数)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3.5 训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, # 训练批次大小，根据显存大小调整，大调大
        gradient_accumulation_steps=8, # 梯度累积，每4个step更新一次参数
        logging_steps=20,
        num_train_epochs=3,
        save_total_limit=2,
        learning_rate=1e-4,
        bf16=True,  # 开启半精度
        gradient_checkpointing=True, # 启用梯度检查点，节省资源
    )

    # 3.6 开始训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds, # 这个是个字典包列表
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True), # 字典内列表转张量
    )
    trainer.train()

    # 3.7 保存适配器 (Adapter)
    trainer.save_model(OUTPUT_DIR + "/final_adapter")
    print(f"✅ LoRA训练完成，适配器已保存至: {OUTPUT_DIR}/final_adapter")

    # --- 4. (可选) 合并模型 ---
    # 注意：如果显存不足，请单独运行此步骤，不要在训练后紧接着运行
    model_to_merge = AutoPeftModelForCausalLM.from_pretrained(
        OUTPUT_DIR + "/final_adapter", device_map={"cuda": 0}, torch_dtype=torch.bfloat16)
    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_DIR + "/merged_model", safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR + "/merged_model")