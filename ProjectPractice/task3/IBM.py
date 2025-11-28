import re
from bs4 import BeautifulSoup
import faiss
from transformers import AutoTokenizer, AutoModel
import torch

# transformers>=4.47.0

# ---------- 1. 清洗数据（完全不动） ----------
# 文本预处理： 读取数据 ，去除HTML标签，并删除文本中所有的电话号码，保存数据。

with open(r'D:\code\python\LearnPyLib\ProjectPractice\task3\3.txt', 'r', encoding="utf-8") as f:
    content = f.read()
data = BeautifulSoup(content, 'html.parser')

# 正则删除数据
clear_tag_data = re.sub(r'<.*?>','', str(data))
clear_phone_data = re.sub(r'电话[:：]?.*', '', clear_tag_data)

with open(r'D:\code\python\LearnPyLib\ProjectPractice\task3\IBM.txt', 'w', encoding="utf-8") as f:
    f.write(clear_phone_data)

# 每行话转换为list
sens = []
for i in clear_phone_data.split("\n"):
    i = i.strip()
    if i:
        sens.append(i)

# ---------- 2. 加载 tokenizer + 模型 ----------
# 用 transforms 加载Embedding 模型bge-base-zh，对处理后数据进行向量化操作，并存入向量库FAISS中，保存数据。

token = AutoTokenizer.from_pretrained(r"D:\code\python\LearnPyLib\model\Em\IBM")
model = AutoModel.from_pretrained(r"D:\code\python\LearnPyLib\model\Em\IBM")
model.eval()

d = model.config.hidden_size # 维度

def proc_sen(sen):
    with torch.no_grad():
        token_data = token(sen, padding=True, return_tensors='pt', max_length=128, truncation=True)
        model_data = model(**token_data)
        return model_data.last_hidden_state[:, 0].cpu().numpy().astype('float32')

# ---------- 3. 创建索引----------

faiss_obj = faiss.IndexFlatL2(d)
for sen in sens:
    tensor_data = proc_sen(sen)
    faiss_obj.add(tensor_data)
faiss.write_index(faiss_obj, r'D:\code\python\LearnPyLib\ProjectPractice\task3\IBM.faiss')

# ---------- 4. 查询 ----------
# 编写代码读取向量库，获取“北京旅游圣地有哪些？”的TOP_3结果。

prompt = "北京旅游圣地有哪些？"
prompt_data = proc_sen(prompt)
faiss_load = faiss.read_index(r'D:\code\python\LearnPyLib\ProjectPractice\task3\IBM.faiss')
D, I = faiss_load.search(prompt_data, 3)
print("D：", D)
print("I：", I)
for i in I[0]:
    print("索引到的数据", sens[i])