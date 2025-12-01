## Transformers
### Pipline（模型任务管线）
```python
from transformers.pipelines import SUPPORTED_TASKS  
from transformers import pipelines, AutoModelForSequenceClassification, AutoTokenizer  
import torch  
  
  
def Look_ST():  
    """  
    简单查看任务分类  
    :return:  
    纯文本（NLP）  
        audio-classification        输入：一段音频（wav/mp3/...）  
        输出：该整段音频属于哪一类（如“枪声”“狗叫”“掌声”）。  
        用例：环境音监控、指令词检测。  
  
        automatic-speech-recognition（ASR）  
        输入：音频 → 输出：对应文本。  
        用例：会议转写、字幕生成。  
  
        text-to-audio（TTS / 音乐 / 音效）  
        输入：文本描述 → 输出：语音、音乐或环境音波形。  
        用例：朗读、提示音、AI 作曲。  
  
        feature-extraction（文本向量）  
        输入：任意文本 → 输出：固定维度的向量。  
        用例：语义搜索、下游聚类、RAG 检索。  
  
        text-classification        输入：句子/段落 → 输出：整段文本的类别标签 + 置信度。  
        用例：情感分析、垃圾邮件识别。  
  
        token-classification        输入：句子 → 输出：每个 token 的标签。  
        用例：命名实体识别（NER）、中文分词、词性标注。  
  
        question-answering（抽取式阅读理解）  
        输入：一段上下文 + 问题 → 输出：答案在上下文中的起止位置。  
        用例：FAQ 自动回答。  
  
        table-question-answering        输入：表格（HTML/CSV）+ 问题 → 输出：答案文本或单元格坐标。  
        用例：财报问答、Excel 对话。  
  
        visual/document-question-answering        输入：图片/扫描件 + 问题 → 输出：文本答案。  
        用例：图表问答、票据字段提取。  
  
        fill-mask        输入：带 <mask>  
        summarization        输入：长文 → 输出：短文摘要。  
        用例：新闻摘要、会议纪要。  
  
        translation        输入：源语言句子 → 输出：目标语言句子。  
        用例：多语种客服、实时字幕。  
  
        text2text-generation（通用 Seq2Seq）  
        输入：任意文本 → 输出：改写/纠错/风格迁移后的文本。  
        用例：拼写纠错、同义改写。  
  
        text-generation（自回归续写）  
        输入：提示语 → 输出：续写内容。  
        用例：故事创作、代码补全。  
  
        zero-shot-classification        输入：文本 + 任意候选标签列表 → 输出：每个标签的概率，无需微调。  
        用例：动态主题分类、冷启动标签。    纯视觉（CV）  
        image-classification        输入：单张图 → 输出：整张图类别。  
        用例：猫狗识别、质量检测。  
  
        zero-shot-image-classification        输入：图 + 任意文本标签列表 → 输出：最匹配的标签，无需再训练。  
        用例：开放集识别、新类别上线。  
  
        image-feature-extraction        输入：图 → 输出：向量。  
        用例：以图搜图、图像聚类。  
  
        image-segmentation        输入：图 → 输出：像素级掩膜（语义/实例/全景）。  
        用例：抠图、自动驾驶可行驶区域。  
  
        image-to-text（图像字幕 / OCR）  
        输入：图 → 输出：自然语言描述或文字串。  
        用例：盲人辅助、截图转文字。  
  
        image-text-to-text（多模态对话）  
        输入：图 + 文本提示 → 输出：文本回答。  
        用例：VQA、图表解释。  
  
        object-detection        输入：图 → 输出：框 + 类别 + 置信度。  
        用例：人脸检测、零售盘点。  
  
        zero-shot-object-detection        输入：图 + 任意文本描述的物体 → 输出：框。  
        用例：新品SKU 无需标注即可检测。  
  
        depth-estimation        输入：单张 RGB → 输出：深度图。  
        用例：AR 测量、机器人避障。  
  
        video-classification        输入：短视频片段 → 输出：动作类别。  
        用例：监控异常行为、体育动作分析。  
  
        mask-generation（SAM 式）  
        输入：图 + 可选提示（点/框/文本）→ 输出：对象掩膜。  
        用例：交互式抠图、标注工具。  
  
        image-to-image        输入：图 → 输出：同尺寸变换后图。  
        用例：超分、去噪、灰度转彩、修复。  
  
        keypoint-matching        输入：两张图 → 输出：对应关键点坐标与匹配。  
        用例：图像对齐、SLAM、全景拼接。  
    音频专用（不含 ASR）  
        zero-shot-audio-classification        输入：音频 + 任意文本标签列表 → 输出：最匹配标签。  
        用例：新声音类别无需重新训练即可上线。  
    """    # print(SUPPORTED_TASKS.items())  
    # 查看任务详情  
    for k, v in SUPPORTED_TASKS.items():  
        print(k, v)  
  
def Create_and_Use_Pipeline():  
    """  
    创建模型，查看模型运行时使用的设备  
    :return:  
    """    pipeline = pipelines.pipeline("text-classification") # 根据任务创建pipline，默认是英文模型，没有会自动拉取  
    # 上述可以指定一些模型，比如支持中文的模型，等等：  
    pipeline = pipelines.pipeline("text-classification", model="在huggingface上复制模型名称")  
    print(pipeline("I'm very happy today")) # 输入文本，返回结果  
    print(pipeline.model.device)  
  
def PreCreate_Model():  
    """  
    预先加载模型，再创建pipline  
    不能只指定模型，而不指定分词器  
    :return:  
    """    model = AutoModelForSequenceClassification.from_pretrained("模型名称") # 预加载模型  
    tokenizer = AutoTokenizer.from_pretrained("模型名称") # 预加载tokenizer  
    pipeline = pipelines.pipeline(  
        "text-classification", model=model, tokenizer=tokenizer  
    ) # 创建pipline  
  
def GPU_Pipeline():  
    """  
    GPU 创建pipline  
    :return:  
    """    pipeline = pipelines.pipeline("text-classification", device=0)  
    print(pipeline("I'm very happy today"))  
  
def Question_Answering_Pipline():  
    """  
    查看pipeline对象的相关属性  
    :return:  
    """    pipeline = pipelines.pipeline("question-answering")  
    print(pipeline) # QuestionAnsweringPipeline类  
    """  
    输入参数如：  
        question (str 或 list[str])        上下文必须搭配出现的“问题”字段。  
        context (str 或 list[str])        给模型阅读的“参考资料”，必须和 question 成对出现。  
                top_k (int，可选，默认 1)        让模型一次性返回几个“最有可能”的答案。  
                doc_stride (int，可选，默认 128)        当“问题 + 上下文”总长度超过模型上限（max_seq_len）时，算法会把上下文切成多段，相邻两段之间重叠多少个 token 就由它决定。  
                max_answer_len (int，可选，默认 15)        模型抽出来的答案最长能有多少个 token（非字符）。响应消息  
                max_seq_len (int，可选，默认 384)        模型一次能处理的“问题 + 上下文”总长上限（token 数）。接收消息  
                max_question_len (int，可选，默认 64)        问题端最长 token 数，超出直接截断尾部。响应消息  
                handle_impossible_answer (bool，可选，默认 False)        是否允许模型输出“无法回答”/“空答案”。  
                align_to_words (bool，可选，默认 True)        后处理阶段是否把模型给出的 token 起止索引“对齐”到真实词语边界。  
    """    print(pipeline(question="问题", context="答案", max_answer_len=15))  
    # 输入问题，输入上下文，返回结果最大个数  
  
def Other_Pipline():  
    """  
    其他模型  
    :return:  
    """    checkpoint = "google/owlvit-base-patch32"  
    detection = pipelines.pipeline(model = checkpoint, task="zero-shot-object-detection")  
    print(detection(  
            "url",  
            ["物体名称", "物体名称"]  
        )  
    )  
  
def Backend_Pipline():  
    """  
    背后原理  
    :return:  
    """    model = AutoModelForSequenceClassification.from_pretrained("模型名称")  # 预加载模型  
    tokenizer = AutoTokenizer.from_pretrained("模型名称")  # 预加载tokenizer  
  
    input_text = "输入的文本"  
    inputs = tokenizer(input_text, return_tensors="pt") # 分词。转为return_tensors，返回pytorch的张量  
    print(inputs) # 输出字典信息，包含输入id、token类型id、注意力掩码  
  
    outputs = model(**inputs) # 等价于model(input_ids, token_type_ids, attention_mask)  
    print(outputs) # 输出预测结果，类型是SequenceClassifierOutput，数据是loss、logits、hidden_states、attentions  
  
    logits = outputs.logits # 拿到预测结果logits，类型是torch.Tensor  
    logits = logits.softmax(dim=1) # 做softmax，实现分类  
    print(logits)  
  
    # 取最大值  
    pred = torch.argmax(logits).item()  
    # 拿到最大值下标  
    print(pred)  
    print(model.config.id2label[pred]) # 拿到最大值对应的标签  
  
if __name__ == '__main__':  
    """  
    预处理Tokenizer——》模型预测Model——》后处理Post Processing   
    """  
    # Look_ST() # 查看任务分类  
    print("----------")  
    # Create_and_Use_Pipeline() # 创建并使用pipeline  
    print("----------")  
    # PreCreate_Model() # 预先加载模型，再创建pipline  
    print("----------")  
    # GPU_Pipeline() # GPU运行  
    print("----------")  
    # Question_Answering_Pipline()  
    print("----------")  
    # Other_Pipline()  
    print("----------")  
    # Backend_Pipline()
```

### Tokenizer（分词器）
```python
from transformers import AutoTokenizer

def Easy_Tokenizer():
    text = "你好"
    tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese") # 预加载tokenizer，模型来自Huggingface
    # 也可以是本地路径
    print(tokenizer) # <tokenizers.models.bert.BertWordPieceTokenizer object at 0x7f9d0a0c0e80>
    # tokenizer.save_pretrained("保存路径") # 保存tokenizer模型

    # 分词
    token = tokenizer.tokenize(text)
    print(token) # ['你', '好']

    # 查看字典
    # print(tokenizer.vocab) # 所有字典数据
    print(tokenizer.vocab_size) # 字典大小

    # 索引转换以便进入神经网络
    ids = tokenizer.encode(text)
    ids_1 = tokenizer.convert_tokens_to_ids(text)
    print("encode", ids) # [101, 872, 1962, 102]
    print("encode", ids_1)
    # 转回来
    token = tokenizer.decode(ids)
    token_1 = tokenizer.convert_ids_to_tokens(ids_1)
    print("decode", token)
    print("decode", token_1)
    # 转成String
    print("转成String", tokenizer.convert_tokens_to_string(token_1))
    # en/decode与convert的区别在于，单个句子中，en/de会有句子开始和句子结束的标记，但是covert没有
    # 可以使用 tokenizer.encode/decode(text, add_special_tokens=False)不适用特殊的标记

    # 填充与截断
    ids = tokenizer.encode(text, max_length=5, truncation=True) # text数据源、最大长度、是否截断
    # 截断会算上句子开始和结束标记
    print(ids)

    # 其他
    attention_mask = tokenizer.get_attention_mask(ids) # 获取一个句子的attention_mask
    # 就是为了区分那部分是句子，那部分是补的0
    print(attention_mask)
    token_type_ids = tokenizer.get_token_type_ids(ids) # 获取一个句子的token_type_ids
    # 就是为了区分是属于哪个句子
    print(token_type_ids)

    # 直接进行超级编码（直接获取所有编码结果）
    ids_plus = tokenizer.encode_plus(text, max_length=5, truncation=True)
    # 返回一个字典，有input_ids、attention_mask、token_type_ids
    # 或者直接
    ids_plus = tokenizer(text, max_length=5, truncation=True)
    # 同样返回一个字典，有input_ids、attention_mask、token_type_ids

    # 批数据处理
    texts = ["你好", "你妈妈"]
    ids = tokenizer(texts, max_length=5, truncation=True)

    # Fast/SlowTokenizer
    # FastTokenizer是使用Rust实现的
    # SlowTokenizer是Python实现的
    print(tokenizer.is_fast) # True

    tokenizer_fast = AutoTokenizer.from_pretrained(
        "uer/roberta-base-finetuned-dianping-chinese", use_fast=True
    ) # 多一个offset_mapping
    input = tokenizer_fast(texts, max_length=5, truncation=True, return_offsets_mapping=True)
    print(input.get("offset_mapping")) # 得到offset_mapping
    """
    例子：
        原文："I love AI"
        分词后：["I", "love", "AI"]
        offset_mapping = [(0,1), (2,6), (7,9)]
        如果模型告诉你“第 2 个 token 是答案”，你就知道答案是原文 2:6 → "love"。
    """
    print(input.word_ids) # 获取单词的索引
    """
    例子：
        原文："ChatGPT is amazing"
        分词后：["Chat", "##G", "##PT", "is", "amazing"]
        word_ids() = [0, 0, 0, 1, 2]
        你就知道前 3 个子词都属于第 0 号单词，后面依次是第 1、2 号单词。
    """
    # 这两个是Fast独有的

    # SlowTokenizer
    tokenizer_slow = AutoTokenizer.from_pretrained(
        "uer/roberta-base-finetuned-dianping-chinese", use_fast=False
    )
    # 有的模型可能不支持FastTokenizer，如果你不支持，那就受着吧

    # 特殊的
    # 有的模型需要在远程进行加载，比如远程加载模型，需要在创建对象的时候加上属性：trust_remote_code=True

if __name__ == "__main__":
    """
    Transformers编码器工具
    """
    Easy_Tokenizer()
```

### EasyModel（模型）
```python
from transformers import (  
    AutoTokenizer,  
    AutoModel,  
    AutoConfig,  
    PretrainedConfig,  
    BertConfig,  
    BertForSequenceClassification,  
    AutoModelForSequenceClassification, # 文本分类任务  
)  
import pandas as pd  
from torch.utils.data import Dataset  
  
  
class MyDataset(Dataset):  
  
    def __init__(self) -> None:  
        super().__init__()  
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")  
        self.data = self.data.dropna()  
  
    def __getitem__(self, index):  
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]  
  
    def __len__(self):  
        return len(self.data)  
  
def Easy_Model():  
    """  
    简单的Model入门  
    :return:  
    """    # 预加载  
    model = AutoModel.from_pretrained("模型名称/模型路径") # 预加载模型  
    config = AutoConfig.from_pretrained("模型名称/模型路径") # 预加载模型配置  
    config.output_attentions = True # 输出注意力，默认为False  
    # PretrainedConfig, BertConfig # 更多配置在这两个类中，其中BertConfig继承PretrainedConfig  
  
    # 模型调用（不带Model Head）  
    prompt = "你好"  
    tokenizer = AutoTokenizer.from_pretrained("模型名称/模型路径")  
    inputs = tokenizer(prompt, return_tensors="pt") # 内置tokenizer编码器  
    print(inputs)  
    # outputs = model(**inputs) # 模型调用，顺带解构数据。得到模型输出  
    # 上述是简单调用  
    # 你需要传入属性进行模型配置修改  
    outputs = model(**inputs, output_attentions=True)  # 模型调用，顺带解构数据。得到模型输出  
    # 输出的Attentions就带有了结果  
    # 取出数据  
    attentions = outputs.attentions # 获取模型输出的Attentions  
    last_hidden_states = outputs.last_hidden_state # 获取模型输出的last_hidden_state  
    print(attentions) # 输出Attentions  
    print(last_hidden_states.size()) # 输出last_hidden_state的维度  
  
    # 模型调用（带Model Head）  
    model = AutoModelForSequenceClassification.from_pretrained("模型名称/模型路径")  
    outputs = model(**inputs) # 模型调用，顺带解构数据。得到模型输出  
    print(outputs)  
    # 相关的属性修改查看：BertForSequenceClassification  
  
    """  
    关于模型带不带Head：  
        不带Head的模型（如BertModel、AutoModel）：  
            只包含骨干网络（Backbone），也就是Transformer的核心架构  
            输出的是隐藏状态（hidden states）或特征表示  
            通常返回的是最后一层的隐藏状态向量
            带Head的模型（如BertForSequenceClassification、AutoModelForSequenceClassification）：  
            包含骨干网络 + 任务特定的头部（Head）  
            在骨干网络基础上添加了针对特定任务的输出层
            直接输出任务相关的结果（如分类logits）  
        它们前置任务类似：
	        输入处理：两者都使用相同的tokenizer进行分词  
            嵌入层：都经过相同的词嵌入、位置嵌入等  
            Transformer编码器：都通过相同的多层Transformer结构  
            特征提取：都得到相同质量的上下文表示        关键差异在输出阶段，Head通常是一个或多个线性层（Linear Layer）  
    """  
if __name__ == "__main__":  
    Easy_Model()
```

### ComModel（模型）
```python
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
  
# -------------------- 可配置参数 --------------------CSV_PATH = "./ChnSentiCorp_htl_all.csv"  
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
  
# -------------------- 自定义数据集 --------------------class MyDataset(Dataset):  
    def __init__(self, csv_path: str):  
        df = pd.read_csv(csv_path).dropna()  
        self.texts = df["text"].tolist()  
        self.labels = df["label"].tolist()  
  
    def __len__(self):  
        return len(self.texts)  
  
    def __getitem__(self, idx):  
        return self.texts[idx], self.labels[idx]  
  
# -------------------- 数据加载函数 --------------------tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  
  
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
  
# -------------------- 训练/验证 Dataloader --------------------dataset = MyDataset(CSV_PATH)  
train_ds, valid_ds = random_split(  
    dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(RANDOM_SEED)  
)  
  
train_loader = DataLoader(  
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn  
)  
valid_loader = DataLoader(  
    valid_ds, batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate_fn  
)  
  
# -------------------- 模型 & 优化器 --------------------model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  
model.to(DEVICE)  
optimizer = Adam(model.parameters(), lr=LR)  
  
# -------------------- 评估函数 --------------------@torch.no_grad()  
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
  
# -------------------- 训练函数 --------------------def train():  
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
  
# -------------------- 单句预测 --------------------def predict(sentence: str):  
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
  
# -------------------- 主入口 --------------------if __name__ == "__main__":  
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
```

### Datasets（数据集）
```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import pandas as pd

def Easy_Datasets_DataLoad():
    # 简单加载普通集合
    datasets = load_dataset("madao33/new-title-chinese")
    print(datasets)
    # train 训练集
    # validation 验证集
    # test 测试集

    # 加载包含子集合的训练任务
    # datasets = load_dataset("super_glue", "boolq") # 加载指定子集合

    # 指定训练/验证/测试集
    # datasets = load_dataset("super_glue", "boolq", split = ["train", "validation", "test"])

    # 加载集合中的指定索引/百分比位置
    # datasets = load_dataset("super_glue", "boolq", split = "train[0:10%]")
    # datasets = load_dataset("super_glue", "boolq", split = "train[0:100]")
    return datasets

def Easy_Opration_Datasets(datasets):
    # 直接进行字典操作
    print(datasets["train"][0]) # 拿到训练集的第一个数据
    # 切片访问
    print(datasets["train"][0:10]) # 拿到训练集的前10个数据
    # 查看某个字段的数据
    print(datasets["train"]["title"][0:5])
    # 查看有哪些字段
    print(datasets["train"].column_names)
    # 类型
    print(datasets["train"].features)

    # 操作数据
    data_trained = datasets["train"]
    data_trained = data_trained.train_test_split(
        test_size=0.2, stratify_by_column="label"
    ) # 划分训练集和测试集, 默认是80%训练集，20%测试集, stratify_by_column="label"是按标签进行划分

    # 选取与过滤
    data_trained_select = data_trained.select(range(10)) # 选取前10个数据，但是返回的是Dataset
    print(data_trained_select)
    # 过滤
    data_trained_filter = data_trained.filter(lambda x: [i for i in range(50)] in x["label"]) # 过滤掉label中50以上的数据
    print(data_trained_filter["title"][0:5]) # 拿到训练集的前5个数据

    # 数据映射
    tokenizer = AutoTokenizer.from_pretrained(
        "uer/roberta-base-finetuned-dianping-chinese"
    ) # 预加载tokenizer，模型来自Huggingface
    def Process_func(examples, tokenizer=tokenizer): # 映射函数，将编码后的数据加载到原来的数据新字段中
        """
        :param examples: 原始数据
        :param tokenizer: 分词器
        :return:
        """
        model_inputs = tokenizer(examples["content"], max_length=512, truncation=True)
        labels = tokenizer(examples["title"], max_length=32, truncation=True)
        # label就是title编码的结果
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    data_trained_map = data_trained.map(Process_func, batched=True) # 映射数据，开启批量映射
    # 如果不是FastTokenizer，那么需要加上属性：num_proc=4，开启多线程进行映射，同时建议将tokenizer作为参数传递到映射函数中
    # 如果你不想要原始字段数据，你可以在调用map函数的时候删除它
    # data_trained_map = data_trained.map(Process_func, batched=True, remove_columns=["title"])

    # 保存与加载
    data_trained_map.save_to_disk("./data")
    data_trained_load = load_dataset("./data")
    # 加载本地数据
    data_trained_load = load_dataset("csv", data_files="./data.csv", split="train")
    # 加载文件类型、路径、是否为DatasetsDict
    # 加上属性：split="train"是train的Dataset，不加是DatasetDict
    # 文件路径上可以写成[]，多个文件
    # 或者使用
    data_trained_load = datasets.from_csv("./data.csv")
    # 直接加载为Dataset
    # 直接加载整个文件夹
    # data_trained_load = load_dataset("csv", data_dir="./", split="train")

    # pandas联动
    data = pd.read_csv("./data.csv")
    DataFrame_to_Dataset = datasets.from_pandas(data) # pandas数据转为Dataset
    Dataset_to_DataFrame = DataFrame_to_Dataset.to_pandas() # Dataset转为pandas数据
    """
    当然有很多from_XXX，比如json、xml、csv等等
    """

    # 自定义加载器，解析复杂的数据结构：参见load_script代码和cmrc2018_trial.json数据
    # 在这里我们只需要加载这个脚本代码就可以
    # data_trained_load = load_dataset("load_script.py", split="train")
    # 但是从 datasets 库的新版开始（>=2.14.0 起），官方已经停止支持直接从 .py 脚本文件加载自定义数据集。

def DataCollator_Dataset():
    dataset = load_dataset("csv",  data_files="./data.csv", split="train")
    # 过滤空数据
    dataset = dataset.filter(lambda x: x["title"] is not None)
    # 数据映射
    tokenizer = AutoTokenizer.from_pretrained(
        "uer/roberta-base-finetuned-dianping-chinese"
    )  # 预加载tokenizer，模型来自Huggingface
    def Process_func(examples): # 映射函数，将编码后的数据加载到原来的数据新字段中
        """
        :param examples: 原始数据
        :param tokenizer: 分词器
        :return:
        """
        model_inputs = tokenizer(examples["content"], max_length=512, truncation=True) # 创建新的变量保存分词
        labels = tokenizer(examples["title"], max_length=32, truncation=True) # 创建新的变量保存分词
        # label就是title编码的结果
        model_inputs["labels"] = labels["input_ids"] # 添加新的字段保存label编码结果
        return model_inputs # 返回新的字段
    data_trained_map = dataset.map(Process_func, batched=True, remove_columns=["title"]) # 映射数据，开启批量映射
    print(data_trained_map[:3]) # 拿到训练集的前3个数据

    # 创建DataCollator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # 创建DataCollator
    # 把“一个 batch 里长度不等的句子”自动补齐（padding）到同一长度。

    # 调用DataLoader：把 Dataset 给出的“一条一条样本”组装成“一个 batch 的张量”。
    train_dataloader = DataLoader(data_trained_map, batch_size=8, collate_fn=data_collator, shuffle=True)
    # 传入
        # 分词器映射函数对象（你的原始数据集）
        # 每轮迭代返回 8 条样本
        # 每个 epoch 都把数据顺序打乱
        # 如何把 8 条样本拼成一个 batch”的自定义函数
    print(train_dataloader) # 可以看到数据变成了Tenser
    # 转换成功！
    # 接下来就是训练数据了


if __name__ == "__main__":
    """
    Transformers数据处理工具
    """
    data = Easy_Datasets_DataLoad()
    Easy_Opration_Datasets(data)

```

### Evaluate（模型评估）
```python
import evaluate
from evaluate.visualization import radar_plot # 可视化雷达图

def Easy_Evaluate():
    """
    简单介绍
    :return:
    """
    # 查看评估工具包支持的模型
    for i in evaluate.list_evaluation_modules():
        print(i)
    # 不想要社区实现的模型评估
    # print(evaluate.list_evaluation_modules(include_community=False))
    # 查看细节
    # print(evaluate.list_evaluation_modules(with_details=True))

    # 加载评估函数
    accuracy = evaluate.load("accuracy") # 在上述打印中有 accuracy
    # 查看函数说明
    print(accuracy.description)
    """
    Accuracy is the proportion of correct predictions among the total number of cases processed. It can be computed with:
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
     Where:
    TP: True positive
    TN: True negative
    FP: False positive
    FN: False negative
    """
    print(accuracy.inputs_description) # 输入参数说明
    """
    Examples:
        Example 1-A simple example
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
            >>> print(results)
            {'accuracy': 0.5}
    
        Example 2-The same as Example 1, except with `normalize` set to `False`.
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], normalize=False)
            >>> print(results)
            {'accuracy': 3.0}
    
        Example 3-The same as Example 1, except with `sample_weight` set.
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4])
            >>> print(results)
            {'accuracy': 0.8778625954198473}
    """
    # 或者直接全部打印
    print(accuracy)

def Algorithm_Evaluate():
    """
    算法评估
    :return:
    """
    # 加载评估函数
    accuracy = evaluate.load("accuracy")
    print("简单的模型评估：", accuracy.compute(
        references=[0, 1, 2, 0, 1, 2],
        predictions=[0, 1, 1, 2, 1, 0],
        sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4]
    )) # 输出计算评估结果，references是预测结果，predictions是真实结果，sample_weight是权重

    # 或者，迭代计算
    for ref, pred in zip([0, 1, 2, 0, 1, 2], [0, 1, 1, 2, 1, 0]):
        accuracy.add(references=ref, predictions=pred)
    print("迭代评估计算：", accuracy.compute())

    # 或者，使用add_batch方法
    for ref, pred in zip([[0,0],[1,1]], [[2,1],[0,2]]):
        accuracy.add_batch(references=ref, predictions=pred) # 不能使用sample_weight
    print("批量评估计算：", accuracy.compute())

def Mult_Algorithm_Evaluate():
    """
    多个评估指标计算
    :return:
    """
    # 加载多个评估函数
    metric = evaluate.combine(evaluations=["accuracy", "precision", "recall", "f1"])
    print(metric)
    # precision_score / recall_score / f1_score 等默认在二分类场景下只返回“正类”的分数，一旦类别数 >2 就不知道该返回哪一类了，于是抛出 ValueError。

    # 简单计算
    print(metric.compute(references=[1, 1, 1, 0, 1, 0],
        predictions=[0, 1, 1, 0, 1, 1]))

def Vision_Evaluate():
    """
    可视化
    """
    data = [
        {'accuracy': 0.95, 'precision': 1.00, 'recall': 0.55, 'f1': 0.71},
        {'accuracy': 0.60, 'precision': 0.65, 'recall': 0.95, 'f1': 0.78},
        {'accuracy': 0.75, 'precision': 0.70, 'recall': 0.60, 'f1': 0.65},
        {'accuracy': 0.50, 'precision': 0.55, 'recall': 0.50, 'f1': 0.52}
    ] # 评估结果
    models = ['model_1', 'model_2', 'model_3', 'model_4'] # 模型名称
    plot = radar_plot(data, models) # 绘制雷达图
    plot.show()

if __name__ == '__main__':
    """
    Transformers模型评估工具包
    """
    # Easy_Evaluate()
    # Algorithm_Evaluate()
    # Mult_Algorithm_Evaluate()
    Vision_Evaluate()
```

### Trainer（模型训练）
```python
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
```