from transformers.pipelines import SUPPORTED_TASKS
from transformers import pipelines, AutoModelForSequenceClassification, AutoTokenizer
import torch


def Look_ST():
    """
    简单查看任务分类
    :return:
    纯文本（NLP）
        audio-classification
        输入：一段音频（wav/mp3/...）
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

        text-classification
        输入：句子/段落 → 输出：整段文本的类别标签 + 置信度。
        用例：情感分析、垃圾邮件识别。

        token-classification
        输入：句子 → 输出：每个 token 的标签。
        用例：命名实体识别（NER）、中文分词、词性标注。

        question-answering（抽取式阅读理解）
        输入：一段上下文 + 问题 → 输出：答案在上下文中的起止位置。
        用例：FAQ 自动回答。

        table-question-answering
        输入：表格（HTML/CSV）+ 问题 → 输出：答案文本或单元格坐标。
        用例：财报问答、Excel 对话。

        visual/document-question-answering
        输入：图片/扫描件 + 问题 → 输出：文本答案。
        用例：图表问答、票据字段提取。

        fill-mask
        输入：带 <mask>

        summarization
        输入：长文 → 输出：短文摘要。
        用例：新闻摘要、会议纪要。

        translation
        输入：源语言句子 → 输出：目标语言句子。
        用例：多语种客服、实时字幕。

        text2text-generation（通用 Seq2Seq）
        输入：任意文本 → 输出：改写/纠错/风格迁移后的文本。
        用例：拼写纠错、同义改写。

        text-generation（自回归续写）
        输入：提示语 → 输出：续写内容。
        用例：故事创作、代码补全。

        zero-shot-classification
        输入：文本 + 任意候选标签列表 → 输出：每个标签的概率，无需微调。
        用例：动态主题分类、冷启动标签。
    纯视觉（CV）
        image-classification
        输入：单张图 → 输出：整张图类别。
        用例：猫狗识别、质量检测。

        zero-shot-image-classification
        输入：图 + 任意文本标签列表 → 输出：最匹配的标签，无需再训练。
        用例：开放集识别、新类别上线。

        image-feature-extraction
        输入：图 → 输出：向量。
        用例：以图搜图、图像聚类。

        image-segmentation
        输入：图 → 输出：像素级掩膜（语义/实例/全景）。
        用例：抠图、自动驾驶可行驶区域。

        image-to-text（图像字幕 / OCR）
        输入：图 → 输出：自然语言描述或文字串。
        用例：盲人辅助、截图转文字。

        image-text-to-text（多模态对话）
        输入：图 + 文本提示 → 输出：文本回答。
        用例：VQA、图表解释。

        object-detection
        输入：图 → 输出：框 + 类别 + 置信度。
        用例：人脸检测、零售盘点。

        zero-shot-object-detection
        输入：图 + 任意文本描述的物体 → 输出：框。
        用例：新品SKU 无需标注即可检测。

        depth-estimation
        输入：单张 RGB → 输出：深度图。
        用例：AR 测量、机器人避障。

        video-classification
        输入：短视频片段 → 输出：动作类别。
        用例：监控异常行为、体育动作分析。

        mask-generation（SAM 式）
        输入：图 + 可选提示（点/框/文本）→ 输出：对象掩膜。
        用例：交互式抠图、标注工具。

        image-to-image
        输入：图 → 输出：同尺寸变换后图。
        用例：超分、去噪、灰度转彩、修复。

        keypoint-matching
        输入：两张图 → 输出：对应关键点坐标与匹配。
        用例：图像对齐、SLAM、全景拼接。
    音频专用（不含 ASR）
        zero-shot-audio-classification
        输入：音频 + 任意文本标签列表 → 输出：最匹配标签。
        用例：新声音类别无需重新训练即可上线。
    """
    # print(SUPPORTED_TASKS.items())
    # 查看任务详情
    for k, v in SUPPORTED_TASKS.items():
        print(k, v)

def Create_and_Use_Pipeline():
    """
    创建模型，查看模型运行时使用的设备
    :return:
    """
    pipeline = pipelines.pipeline("text-classification") # 根据任务创建pipline，默认是英文模型，没有会自动拉取
    # 上述可以指定一些模型，比如支持中文的模型，等等：
    pipeline = pipelines.pipeline("text-classification", model="在huggingface上复制模型名称")
    print(pipeline("I'm very happy today")) # 输入文本，返回结果
    print(pipeline.model.device)

def PreCreate_Model():
    """
    预先加载模型，再创建pipline
    不能只指定模型，而不指定分词器
    :return:
    """
    model = AutoModelForSequenceClassification.from_pretrained("模型名称") # 预加载模型
    tokenizer = AutoTokenizer.from_pretrained("模型名称") # 预加载tokenizer
    pipeline = pipelines.pipeline(
        "text-classification", model=model, tokenizer=tokenizer
    ) # 创建pipline
    """
    AutoTokenizer.from_pretrained("bert-base-chinese")
        ├─ 读 tokenizer_config.json → 知道类名
        ├─ 读 vocab.txt / tokenizer.json → 得到 id↔token 映射
        └─ 返回 tok 对象，用来 encode / decode

    AutoModel.from_pretrained("bert-base-chinese")
        ├─ 读 config.json → 建一个空壳 BertModel
        └─ 读 pytorch_model.bin → 把权重填进去，返回可用模型
    """

def GPU_Pipeline():
    """
    GPU 创建pipline
    :return:
    """
    pipeline = pipelines.pipeline("text-classification", device=0)
    print(pipeline("I'm very happy today"))

def Question_Answering_Pipline():
    """
    查看pipeline对象的相关属性
    :return:
    """
    pipeline = pipelines.pipeline("question-answering")
    print(pipeline) # QuestionAnsweringPipeline类
    """
    输入参数如：
        question (str 或 list[str])
        上下文必须搭配出现的“问题”字段。
        
        context (str 或 list[str])
        给模型阅读的“参考资料”，必须和 question 成对出现。
        
        top_k (int，可选，默认 1)
        让模型一次性返回几个“最有可能”的答案。
        
        doc_stride (int，可选，默认 128)
        当“问题 + 上下文”总长度超过模型上限（max_seq_len）时，算法会把上下文切成多段，相邻两段之间重叠多少个 token 就由它决定。
        
        max_answer_len (int，可选，默认 15)
        模型抽出来的答案最长能有多少个 token（非字符）。响应消息
        
        max_seq_len (int，可选，默认 384)
        模型一次能处理的“问题 + 上下文”总长上限（token 数）。接收消息
        
        max_question_len (int，可选，默认 64)
        问题端最长 token 数，超出直接截断尾部。响应消息
        
        handle_impossible_answer (bool，可选，默认 False)
        是否允许模型输出“无法回答”/“空答案”。
        
        align_to_words (bool，可选，默认 True)
        后处理阶段是否把模型给出的 token 起止索引“对齐”到真实词语边界。
    """
    print(pipeline(question="问题", context="答案", max_answer_len=15))
    # 输入问题，输入上下文，返回结果最大个数

def Other_Pipline():
    """
    其他模型
    :return:
    """
    checkpoint = "google/owlvit-base-patch32"
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
    """
    model = AutoModelForSequenceClassification.from_pretrained("模型名称")  # 预加载模型
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
    Transformers任务工具
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
