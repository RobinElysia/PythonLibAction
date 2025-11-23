# -*- coding: utf-8 -*-
"""
中文情感分类示例（ChnSentiCorp_htl_all.csv）
依赖：pandas, torch, transformers, scikit-learn
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from torch.optim import Adam

# -------------------- 可配置参数 --------------------
CSV_PATH = "./ChnSentiCorp_htl_all.csv"
MODEL_NAME = "hfl/rbt3"
MAX_LEN = 128
BATCH_SIZE = 32
EVAL_BATCH = 64
LR = 2e-5
EPOCHS = 3
LOG_STEP = 100
RANDOM_SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------

# 固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# -------------------- 自定义数据集 --------------------
class MyDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path).dropna()
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# -------------------- 数据加载函数 --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def collate_fn(batch):
    texts, labels = zip(*batch)
    enc = tokenizer(
        list(texts),
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    return enc

# -------------------- 训练/验证 Dataloader --------------------
dataset = MyDataset(CSV_PATH)
train_ds, valid_ds = random_split(
    dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(RANDOM_SEED)
)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_ds, batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate_fn
)

# -------------------- 模型 & 优化器 --------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)

# -------------------- 评估函数 --------------------
@torch.no_grad()
def evaluate():
    model.eval()
    total_correct, total_num = 0, 0
    for batch in valid_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_num += batch["labels"].size(0)
    return total_correct / total_num

# -------------------- 训练函数 --------------------
def train():
    global_step, total_loss, step_count = 0, 0.0, 0
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step_count += 1
            global_step += 1

            if global_step % LOG_STEP == 0:
                avg_loss = total_loss / step_count
                print(f"Epoch {epoch} | Step {global_step} | Loss {avg_loss:.4f}")

        # 每个 epoch 结束评估一次
        acc = evaluate()
        print(f"Epoch {epoch} | Val Acc {acc:.4f}")

# -------------------- 单句预测 --------------------
def predict(sentence: str):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            sentence,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
    id2label = {0: "差评！", 1: "好评！"}
    return id2label[pred]

# -------------------- 主入口 --------------------
if __name__ == "__main__":
    """
    模型完整流程
    """
    train()  # 训练

    # 快速测试
    test_sen = "我觉得这家酒店不错，饭很好吃！"
    print("输入：", test_sen)
    print("预测：", predict(test_sen))

    # 导出 pipeline（可选）
    model.config.id2label = {0: "差评！", 1: "好评！"}
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE.type == "cuda" else -1,
    )
    print(pipe(test_sen))