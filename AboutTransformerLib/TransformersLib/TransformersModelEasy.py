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
    """
    # 预加载
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
            特征提取：都得到相同质量的上下文表示
        关键差异在输出阶段，Head通常是一个或多个线性层（Linear Layer）
    """

if __name__ == "__main__":
    """
    Transformers模型对象的简单入门
    """
    Easy_Model()