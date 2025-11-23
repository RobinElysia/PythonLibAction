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
