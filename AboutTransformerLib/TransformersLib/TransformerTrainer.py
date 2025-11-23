# 一些自带的模型是支持的，但是对于魔改的输入输出模型是不支持的
# ---------------------------------------------------------
# 一个可运行的 Trainer + TrainingArguments 示例
# 任务：使用 DistilBERT 在 IMDB 数据集上做情感分类
# ---------------------------------------------------------
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ---------------------------------------------------------
# 1. 加载数据集
# ---------------------------------------------------------
# HuggingFace 会自动下载 IMDB 数据集（训练 25k，测试 25k）
dataset = load_dataset("imdb")

# 数据集字段为：{"text": ..., "label": 0/1}


# ---------------------------------------------------------
# 2. 加载 Tokenizer
# ---------------------------------------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# ---------------------------------------------------------
# 3. 编写 tokenize 函数
# ---------------------------------------------------------
def tokenize_fn(example):
    """
    对每条样本进行 tokenization
    truncation=True 会自动截断过长文本
    padding=True 在批处理时让长度对齐（dynamic padding）
    """
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",  # 或 dynamic padding: padding="longest"
        max_length=256, # 减小，则可以降低显存占用和时间
    )


# 将 tokenize 函数应用到整个数据集
tokenized_ds = dataset.map(tokenize_fn, batched=True)

# Trainer 要求输入中必须包含 input_ids / attention_mask / labels
tokenized_ds = tokenized_ds.rename_column("label", "labels")

# 使数据集只保留必要字段
tokenized_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)


# ---------------------------------------------------------
# 4. 加载预训练模型（分类头自动加载）
# ---------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # IMDB 是二分类任务
)


# ---------------------------------------------------------
# 5. 定义评价指标
# ---------------------------------------------------------
def compute_metrics(eval_pred):
    """
    eval_pred 包含两个元素：
    - logits（模型输出）
    - labels（真实标签）
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    prec, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------
# 6. 创建 TrainingArguments
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./imdb_distilbert",  # 模型保存位置

    # === 训练参数 ===
    num_train_epochs=1,              # 演示用 1 epoch（实际可增大）
    per_device_train_batch_size=4,  # 每块 GPU 上的 batch
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=32,   # 32 个 step 累积梯度
    # 可将 gradient_accumulation_steps 设置为 32
    # per_device_eval_batch_size 和 per_device_train_batch_size 改为 1
    # 这个时候显存会减少，但训练时间会大幅增加

    gradient_checkpointing=True, # 启用 gradient checkpointing
    # 可以降低一点显存占用，但是不如上述来的快，但是时间会更大

    # 也可以指定优化器，切换压力较小的优化器
    optim="adafactor", # 默认为 AdamW，切换可以在OptimizerNames看

    # === 优化器相关 ===
    learning_rate=5e-5,
    weight_decay=0.01,

    # === 日志 / 保存 ===
    logging_steps=50,                # 每 50 step 打印日志
    evaluation_strategy="epoch",     # 每个 epoch 结束时 eval
    save_strategy="epoch",           # 每个 epoch 保存 checkpoint
    save_total_limit=2,              # 最多保留2个checkpoint

    # === 其他设置 ===
    load_best_model_at_end=True,     # 根据指标自动加载最佳模型
    metric_for_best_model="accuracy",# 评价指标

    fp16=True,                       # 如果有 GPU，则开启混合精度
)

# ---------------------------------------------------------
# 7. 冻结层
# ---------------------------------------------------------
# 遍历模型的所有参数，将名称中不包含"classifier"的参数设置为不可训练状态（冻结参数）。
# 大幅减少时间和显存占用
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

# ---------------------------------------------------------
# 8. 创建 Trainer
# ---------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,   # 自定义评估
)


# ---------------------------------------------------------
# 9. 开始训练
# ---------------------------------------------------------
print(trainer.train())

# ---------------------------------------------------------
# 10. 评估模型
# ---------------------------------------------------------
print(trainer.evaluate(tokenized_ds["test"]))

# ---------------------------------------------------------
# 11. 模型预测
# ---------------------------------------------------------
print(trainer.predict(tokenized_ds["test"]))