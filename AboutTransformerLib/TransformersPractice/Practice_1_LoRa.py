import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    AutoPeftModelForCausalLM
)

# --- 1. 配置常量 ---
# 模型路径（通常是本地下载好的HuggingFace格式模型）
MODEL_PATH = "/opt/model/gemma-2-2b"
# 训练数据路径（JSON格式指令集）
DATA_PATH = "/opt/data/lora.json"
# 输出路径
OUTPUT_DIR = "./gemma2_lora_finetune_output"

# --- 2. 定义Prompt模板 ---
# 这是Gemma模型特有的对话模板格式，包含 system, user, assistant 标签
PROMPT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "你是一个有用的助手，擅长根据用户提问生成信息，包括虚构内容。<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}{input}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

# --- 3. 初始化分词器 ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
# 补全token如果没有定义，则使用结束token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --- 4. 数据预处理函数 (核心) ---
def process_func(example):
    """
    example 是一个原始数据样本，代表从数据集中传入的单条未处理数据记录
        它通常是一个字典(dict)结构，包含原始的文本字段，如:
            instruction: 指令文本
            input: 输入内容(可选)
            output: 目标输出文本
    将原始数据转换为模型输入的 token id，并设置 label masking
    为什么设置label mask（让模型在训练时更加专注地学习从输入到输出的映射关系，提高训练效率和最终效果）。
    目的：让模型只计算回答部分（Response）的损失，而不计算问题部分（Instruction）的损失。
    """
    MAX_LENGTH = 1024 # 输入序列的最大长度限制。

    # A. 构建输入文本（Instruction指令部分）
    instruction_text = PROMPT_TEMPLATE.format( # 这部分在上述提示词中有占位符，所以这里直接使用了format进行了替换
        instruction=example['instruction'],
        input=example.get('input', '')  # 兼容input为空的情况
    )

    # B. 构建回答文本（Response部分）
    # 加上 <|eot_id|> 表示生成结束
    response_text = f"{example['output']}<|eot_id|>"

    # C. 分词
    # add_special_tokens=False 因为模板里已经手动加了特殊token
    instruction_ids = tokenizer(instruction_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    # D. 拼接 Input IDs
    # 加上 pad_token_id 是为了处理可能的对齐问题，通常在DataCollator中处理padding，这里是手动拼接逻辑
    input_ids = instruction_ids + response_ids + [tokenizer.pad_token_id]

    # E. 构建 Attention Mask (1表示关注，0表示忽略)
    # 这里的 + [1] 对应上面的 pad_token_id
    attention_mask = [1] * len(instruction_ids) + [1] * len(response_ids) + [1]

    # F. 构建 Labels (关键点)
    # Instruction部分设为 -100，PyTorch的CrossEntropyLoss会自动忽略 -100 的位置
    # 只有 Response 部分参与 Loss 计算
    labels = [-100] * len(instruction_ids) + response_ids + [tokenizer.pad_token_id]

    # G. 截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# --- 5. 主流程 ---
if __name__ == "__main__":

    # 5.1 加载数据集
    print("🚀 1. 正在加载和处理数据集...")
    df = pd.read_json(DATA_PATH)
    ds = Dataset.from_pandas(df) # 将pandas数据转化为Huggingface家族Dataset数据集
    # 使用 map 批量处理数据，remove_columns 删除原始文本列，只保留 tensor
    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
    # map的作用是：批量处理数据，将数据映射为模型输入的格式。
    # process_func函数是数据处理的自定义函数，将原始数据处理为模型输入的格式。
    # remove_columns删除原始文本列，只保留 tensor

    # 5.2 加载模型
    print("\n🚀 2. 正在加载Gemma2模型...")
    # torch_dtype=torch.bfloat16 是为了节省显存并保持精度（Ampere架构显卡推荐）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # 开启梯度检查点，节省显存（用计算换显存）
    model.enable_input_require_grads()

    # 5.3 冻结前6层权重 (题目特定要求)
    # 虽然使用了LoRA，但题目要求显式冻结基础模型的前几层，可能是为了演示层级操作
    print("\n🚀 3. 正在冻结模型前6层权重...")
    for param in model.parameters():
        param.requires_grad = False  # 先冻结所有

    # 这里的逻辑是：如果找到 layers 属性，则再次确认前6层是冻结的
    # 实际上，使用LoRA时，基础模型通常全是冻结的。这里的代码可能是为了确保某些特殊层不参与计算。
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers_to_freeze = model.model.layers[:6]
        for layer in layers_to_freeze:
            for p in layer.parameters():
                p.requires_grad = False

    # 5.4 配置 LoRA
    print("\n🚀 4. 正在配置LoRA...")
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        # 目标模块：通常是 Attention 层的投影矩阵
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=8,  # LoRA 秩，决定了可训练参数量的大小
        lora_alpha=32,  # 缩放系数，通常是 r 的 2-4 倍
        lora_dropout=0.1
    )

    # 将 LoRA 适配器挂载到基础模型上
    model = get_peft_model(model, config)

    # 确保 LoRA 参数是可训练的 (requires_grad = True)
    for name, param in model.named_parameters():
        if "lora" in name or "LoRA" in name:
            param.requires_grad = True

    model.print_trainable_parameters()  # 打印可训练参数比例

    # 5.5 配置训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,  # 显存小则调小
        gradient_accumulation_steps=4,  # 梯度累积，变相增大 batch size
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,  # LoRA通常比全量微调学习率大一点
        save_on_each_node=True,
        gradient_checkpointing=True,
        bf16=True,  # 开启半精度
        fp16=False,
        save_total_limit=1  # 只保留最近的一个模型
    )

    # 5.6 开始训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        # DataCollator 负责将 batch 数据动态 padding 到最大长度
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    print("\n🚀 正在开始LoRA微调...")
    trainer.train()

    # 5.7 保存 Adapter
    final_adapter_path = f"{OUTPUT_DIR}/final_adapter"
    trainer.model.save_pretrained(final_adapter_path)
    print(f"✅ LoRA适配器权重已保存到: {final_adapter_path}")

    # 5.8 合并权重 (Merge LoRA into Base Model)
    # 推理时，为了速度，通常将 LoRA 权重合并回基础模型
    print("\n🚀 5. 正在合并LoRA权重到原始模型...")
    merged_model_path = f"{OUTPUT_DIR}/merged_model"
    os.makedirs(merged_model_path, exist_ok=True)

    # 重新加载 Base + Adapter
    model_to_merge = AutoPeftModelForCausalLM.from_pretrained(
        final_adapter_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    # 执行合并卸载操作
    merged_model = model_to_merge.merge_and_unload()

    # 保存完整模型
    merged_model.save_pretrained(merged_model_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_path)
    print("--- 训练和合并流程结束 ---")