from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup
import faiss
from torch import no_grad
import re

# ---------- 1. 清洗数据（完全不动） ----------
# 文本预处理： 读取数据 ，去除HTML标签，并删除文本中所有的电话号码，保存数据。

# 导入时数据
with open(r'D:\code\python\LearnPyLib\ProjectPractice\task3\3.txt', 'r', encoding='utf-8') as f:
    content = f.read()
# 汤处理
data_load = BeautifulSoup(content, 'html.parser')
# 正则删除
data_clear_tag = re.sub(r'<.*?>', '', str(data_load))
data_clear_phone = re.sub(r'电话[:：]?.*', '', data_clear_tag)
# 数据转换
sens = []
for i in data_clear_phone.split('\n'):
    i = i.strip()
    if i:
        sens.append(i)
# 数据保存
with open(r'D:\code\python\LearnPyLib\ProjectPractice\task3\jina.txt', 'w', encoding='utf-8') as f:
    f.write(data_clear_phone)

# ---------- 2. 加载 tokenizer + 模型 ----------
# 用 transforms 加载Embedding 模型bge-base-zh，对处理后数据进行向量化操作，并存入向量库FAISS中，保存数据。

# 创建模型
tok = AutoTokenizer.from_pretrained(r'D:\code\python\LearnPyLib\model\Em\jina', trust_remote_code=True)
model = AutoModel.from_pretrained(r'D:\code\python\LearnPyLib\model\Em\jina', trust_remote_code=True)
# 开启模型eval
model.eval()
# 定义维度
d = model.config.hidden_size
# 定义函数
def proc_sen(sen):
    with no_grad:
        token_data = tok(sen, padding=True, return_tensors='pt', max_length=128)
        model_data = model(**token_data)
        return model_data.last_hidden_state[:, 0].cpu().numpy().astype('float32')
# 迭代列表
# 创建faiss对象
faiss_ = faiss.IndexFlatL2(d)
for i in sens:
    ten_data = proc_sen(i)
    faiss_.add(ten_data)
faiss.write_index(faiss_, r'D:\code\python\LearnPyLib\ProjectPractice\task3\jina.faiss')

# ---------- 3. 创建索引----------

faiss_load = faiss.read_index(r'D:\code\python\LearnPyLib\ProjectPractice\task3\jina.faiss')

# ---------- 4. 查询 ----------
# 编写代码读取向量库，获取“北京旅游圣地有哪些？”的TOP_3结果。
prompt = "北京旅游圣地有哪些？"
ten_pro = proc_sen(prompt)
D, I = faiss_load.search(ten_pro, 3)
print("D：", D)
print("I：", I)
for i in I[0]:
    print(sens[i])