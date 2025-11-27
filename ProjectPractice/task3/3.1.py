import os
# 设置环境变量，允许 OpenMP 重复加载
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import faiss
import re
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel


# ---------- 1. 清洗数据 ----------
with open(r"/ProjectPractice/task3/3.txt", 'r', encoding='utf-8') as f:
    content = f.read()
soup = BeautifulSoup(content, 'html.parser')
text = re.sub(r'<.*?>', '', str(soup)) # 替换为空
text = re.sub(r'电话[:：]?.*', '', text) # 替换为空
lines = []                 # 用来存非空行
for raw_line in text.split('\n'):   # 每一行进行遍历
    clean_line = raw_line.strip()   # 去掉前后空白
    if clean_line:                  # 如果不是空字符串
        lines.append(clean_line)    # 留下这一行
# 文本按行拆开，去掉每行前后的空白字符，并扔掉所有空行，最后得到一个只含非空行且首尾无空白的列表。
# ['第1行', '第2行', '第3行']

# ---------- 2. 加载 tokenizer + 模型 ----------
tok = AutoTokenizer.from_pretrained(r"D:\model\Em\bge")
model = AutoModel.from_pretrained(r"D:\model\Em\bge")
print("模型维度：", model.config.hidden_size) # 查看维度
model.eval()          # 推理模式
"""
做 向量抽取 / 推理 / 评测 时，务必 model.eval()；
做 训练 / 微调 时，再 model.train() 切回去即可。
此处建议开启。
"""

def cls_vec(s_data):       # 传单句进来，单句→768维向量
    with torch.no_grad(): # 上下文管理器，禁用梯度计算，减少内存消耗并加速推理。
        tokenizer_trans_tensors = tok(s_data, return_tensors='pt', max_length=128, truncation=True) # 分词后的数据
        out = model(**tokenizer_trans_tensors) # 将分词结果解包传递给模型，进行前向传播。

        # 3. 取 [CLS] 位置向量（第 0 个 token）并转到 CPU
        cls_embedding = out.last_hidden_state[:, 0] # (1, 768)
        # [:, 0]代表:取所有第一维度，第一维度我全要，第二维度取第一个，第三维度不写代表全要
        # 等价于[:, 0, :]
        cls_embedding = cls_embedding.cpu()  # shape: (768,)
        # 放在cpu上计算

        # 4. 转为 float32 numpy 数组，PyTorch 张量 → Numpy 数组，再把精度从 float64 降到 float32，内存直接砍半。
        return cls_embedding.numpy().astype('float32')

# ---------- 3. 建索引（维度固定 768） ----------
dim = model.config.hidden_size
faiss_index = faiss.IndexFlatL2(dim)
for sent in lines:
    v = cls_vec(sent) # 调用函数，返回一个 768维向量
    faiss_index.add(v)          # 每次加 1×768
faiss.write_index(faiss_index, r"D:\code\python\LearnPyLib\ProjectPractice\3.faiss")

# ---------- 4. 查询 ----------
prompt = "上海旅游胜地有哪些？"
faiss_loaded = faiss.read_index(r"D:\code\python\LearnPyLib\ProjectPractice\3.faiss")
q = cls_vec(prompt)
D, I = faiss_loaded.search(q, 3)
print("Top3 索引:", I)
print("距离:", D)
for i in I[0]:
    print("文本:", lines[i])