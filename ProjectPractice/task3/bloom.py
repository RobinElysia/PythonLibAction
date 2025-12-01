import os
# 设置环境变量，允许 OpenMP 重复加载
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import re

# ---------- 1. 清洗数据（完全不动） ----------
# 文本预处理： 读取数据 ，去除HTML标签，并删除文本中所有的电话号码，保存数据。

with open(r"D:\code\python\LearnPyLib\ProjectPractice\task3\3.txt", 'r', encoding='utf-8') as f:
    content = f.read()

clear_tag_data = re.sub(r'<.*?>', '', content) # 删除HTML标签
clear_number_data = re.sub(r'电话[:：]?.*', '', clear_tag_data) # 删除电话号码

with open(r'D:\code\python\LearnPyLib\ProjectPractice\bloom.txt', 'w', encoding='utf-8') as f:
    f.write(clear_number_data)

sens = []
for i in clear_number_data.split("\n"):
    i = i.strip()
    if i:
        sens.append(i)

# ---------- 2. 加载 tokenizer + 模型 ----------
# 用 transforms 加载Embedding 模型bge-base-zh，对处理后数据进行向量化操作，并存入向量库FAISS中，保存数据。

token = AutoTokenizer.from_pretrained(r"D:\code\python\LearnPyLib\model\Em\bloom") # 我的模型在本地，你们需要更改
model = AutoModel.from_pretrained(r"D:\code\python\LearnPyLib\model\Em\bloom")
model.eval()
d = model.config.hidden_size # 1536维度

def proc_sen_to_tensor(sen):
    with torch.no_grad():  # 上下文管理器，禁用梯度计算，减少内存消耗并加速推理。
        token_data_sen = token(sen, return_tensors='pt', max_length=128, truncation=True)
        out = model(**token_data_sen)
        print(out.last_hidden_state[:, 0])
        return out.last_hidden_state[:, 0].cpu().numpy().astype('float32')

# ---------- 3. 创建索引----------

faiss_obj = faiss.IndexFlatL2(d)
for sen in sens:
    array_data = proc_sen_to_tensor(sen)
    faiss_obj.add(array_data)
faiss.write_index(faiss_obj, r"D:\code\python\LearnPyLib\ProjectPractice\3.2.faiss") # 保存向量库

# ---------- 4. 查询 ----------
# 编写代码读取向量库，获取“北京旅游圣地有哪些？”的TOP_3结果。

prompt = "北京旅游圣地有哪些？"
faiss_obj = faiss.read_index(r"D:\code\python\LearnPyLib\ProjectPractice\3.2.faiss")
query_vec = proc_sen_to_tensor(prompt)
D, I = faiss_obj.search(query_vec, 3)
print("Top3 索引:", I)
print("距离:", D)
for i in I[0]:
    print("文本:", sens[i])