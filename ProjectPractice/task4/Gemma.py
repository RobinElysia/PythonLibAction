# 读取/opt/data/lora.json数据，将数据处理为lora微调所需格式
# 编写代码加载Gemma2模型的预训练权重
# 编写模型微调代码，冻结前6层权重。
# 设定微调参数，设置r=8, lora_alpha=32, lora_dropout=0.1。
# 合并LoRA微调后的权重到原始模型中。
# 编写终端对话代码，验证模型对于“北京地摊文化博物馆”、“新疆海洋文化博物馆”等虚构旅游地的信息输出能力。

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
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "你是一个有用的助手。<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}{input}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


# --- 2. 数据处理函数 ---
def process_func(example):
    MAX_LENGTH = 1024

    # 构建文本
    instruction = PROMPT_TEMPLATE.format(instruction=example['instruction'], input=example.get('input', ''))
    response = f"{example['output']}<|eot_id|>"

    # 分词 (add_special_tokens=False 因为模板已包含)
    ins_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    res_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    # 拼接、补全与截断
    input_ids = (ins_ids + res_ids + [tokenizer.pad_token_id])[:MAX_LENGTH]
    attention_mask = ([1] * len(input_ids))[:MAX_LENGTH]
    # 创建一个长度等于 input_ids 的列表，所有元素都是 1

    # 构建 Labels (指令部分设为 -100 不计算loss)
    labels = ([-100] * len(ins_ids) + res_ids + [tokenizer.pad_token_id])[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# --- 3. 主流程 ---
if __name__ == "__main__":
    # 3.1 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 3.2 准备数据
    ds = Dataset.from_pandas(pd.read_json(DATA_PATH))
    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

    # 3.3 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"":0},
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
        per_device_train_batch_size=1,  # 从4减小到1
        gradient_accumulation_steps=16,  # 从4增加到16以维持有效批次大小
        logging_steps=10,
        num_train_epochs=1,  # 从3减少到1以加快训练
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,  # 从2减少到1以节省存储空间
        learning_rate=1e-4,
        fp16=True,  # 使用fp16替代bf16（更兼容性）
        gradient_checkpointing=True,
        dataloader_pin_memory=False
        # output_dir=OUTPUT_DIR,
        # per_device_train_batch_size=4,
        # gradient_accumulation_steps=4,
        # logging_steps=10,
        # num_train_epochs=3,
        # save_strategy="steps",
        # save_steps=100,
        # save_total_limit=2,
        # learning_rate=1e-4,
        # bf16=True,  # 开启半精度
        # gradient_checkpointing=True, # 启用梯度检查点，节省资源
    )

    # 3.6 开始训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    # 3.7 保存适配器 (Adapter)
    trainer.save_model(OUTPUT_DIR + "/final_adapter")
    print(f"✅ LoRA训练完成，适配器已保存至: {OUTPUT_DIR}/final_adapter")

    # --- 4. (可选) 合并模型 ---
    # 注意：如果显存不足，请单独运行此步骤，不要在训练后紧接着运行
    model_to_merge = AutoPeftModelForCausalLM.from_pretrained(
        OUTPUT_DIR + "/final_adapter", device_map={"": 0}, torch_dtype=torch.bfloat16)
    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_DIR + "/merged_model", safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR + "/merged_model")