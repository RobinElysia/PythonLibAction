AutoTokenizer.from_pretrained("bert-base-chinese")
    ├─ 读 tokenizer_config.json → 知道类名
    ├─ 读 vocab.txt / tokenizer.json → 得到 id↔token 映射
    └─ 返回 tok 对象，用来 encode / decode

AutoModel.from_pretrained("bert-base-chinese")
    ├─ 读 config.json → 建一个空壳 BertModel
    └─ 读 pytorch_model.bin → 把权重填进去，返回可用模型