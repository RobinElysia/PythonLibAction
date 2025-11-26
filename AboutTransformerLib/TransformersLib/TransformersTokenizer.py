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