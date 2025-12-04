我会以Gemma2:2b和Qwen为例子进行微调，废话少说，直接开始。

## 提示词模板参考

| 模型家族                  | system 支持                                      | 角色名                    | 典型边界标记                                                 | 官方模板片段（Jinja2）                                       | 备注                                                   |
| :------------------------ | :----------------------------------------------- | :------------------------ | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------- |
| **Qwen2 / Qwen-VL**       | ✅ 默认带  <br>`You are a helpful assistant`      | user / assistant / system | `<|im_start|>role\n...<|im_end|>\n`                          | `{% for message in messages %}{{'<|im_start|>'+message['role']+'\n'+message['content']+'<|im_end|>'+'\n'}}{% endfor %}` | 多模态时插入 `<|vision_start|>...<|vision_end|>`       |
| **ChatML（OpenAI 风格）** | ✅ 推荐显式给出                                   | user / assistant / system | 同上                                                         | 官方单行版：  <br>`{%- for message in messages %}{{'<|im_start|>'+message['role']+'\n'+message['content']+'<|im_end|>'+'\n'}}{% endfor %}` | 被多数新模型直接复用                                   |
| **Llama-2-Chat**          | ✅ 用 `<<SYS>>\n...\n<</SYS>>` 嵌在第一条 user 里 | user / assistant          | `<s>[INST] <<SYS>>...\n<</SYS>>\n{user} [/INST] {assistant} </s><s>[INST] {user} [/INST]` | 见下方完整模板                                               | 官方**不允许**单独 system 消息，必须嵌在首条 user 内部 |
| **Llama-3-Instruct**      | ✅ 独立 system 消息                               | user / assistant / system | `<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>` | `{% for message in messages %}{{'<|start_header_id|>'+message['role']+'<|end_header_id|>\n\n'+message['content']+'<|eot_id|>'}}{% endfor %}` | 与 llama-2 不兼容                                      |
| **Mistral-Instruct v0.x** | ❌ 官方模板**显式禁止 system**                    | user / assistant          | `<s>[INST] {user} [/INST] {assistant}</s>`                   | `{% if messages[0]['role']=='system' %}{{ raise_exception('System role not supported') }}{% endif %}...` | 想加 system 需自己改模板                               |
| **Gemma / Gemma-2**       | ❌ 官方**无 system**                              | user / model              | `<bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>` | 模板里把 assistant 重命名成 model                            |                                                        |
| **Yi-Chat**               | ✅ 可选                                           | user / assistant / system | 同 ChatML                                                    | 与 Qwen 相同格式，官方无默认 system                          |                                                        |
| **DeepSeek-Chat**         | ✅ 默认 system                                    | user / assistant / system | 同 ChatML                                                    | 直接复用 Qwen 模板                                           |                                                        |
| **GLM-4-Chat**            | ✅                                                | user / assistant / system | `[gMASK]sop<|user|>\n...<|assistant|>\n...<|system|>\n...`   | 官方模板较长，见仓库 tokenizer_config.json                   | 含 `gMASK`+`sop` 前缀                                  |
| **Baichuan2-Chat**        | ✅                                                | user / assistant / system | `<reserved_102>{user}<reserved_103>{assistant}<reserved_102>...` | 用数字 ID 作角色标记                                         | 需关 `add_bos_token=False`                             |

### 使用

#### Qwen2 / Qwen-VL

```python
messages = [
    {"role": "system", "content": "你是一个有用的助手"},
    {"role": "user", "content": "请解释一下机器学习"},
    {"role": "assistant", "content": "机器学习是人工智能的一个分支..."},
    {"role": "user", "content": "具体说说监督学习"}
]

formatted = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
# 输出：<|im_start|>system\n你是一个有用的助手<|im_end|>\n<|im_start|>user\n请解释一下机器学习<|im_end|>\n<|im_start|>assistant\n机器学习是人工智能的一个分支...<|im_end|>\n<|im_start|>user\n具体说说监督学习<|im_end|>\n<|im_start|>assistant
```

#### ChatML（OpenAI风格）

```python
messages = [
    {"role": "system", "content": "你是一个AI助手"},
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
    {"role": "user", "content": "今天天气怎么样"}
]

formatted = tokenizer.apply_chat_template(
    messages, 
    tokenize=False,
    add_generation_prompt=True
)
# 输出：<|im_start|>system\n你是一个AI助手<|im_end|>\n<|im_start|>user\n你好！<|im_end|>\n<|im_start|>assistant\n你好！有什么可以帮助你的？<|im_end|>\n<|im_start|>user\n今天天气怎么样<|im_end|>\n<|im_start|>assistant
```

#### Llama-2-Chat

```python
messages = [
    {"role": "user", "content": "什么是人工智能？"}
]

# 带系统消息的方式（必须嵌入在第一条user中）
messages_with_system = [
    {
        "role": "user", 
        "content": "<<SYS>>\n你是一个AI专家助手<</SYS>>\n什么是人工智能？"
    }
]

formatted = tokenizer.apply_chat_template(
    messages_with_system,
    tokenize=False,
    add_generation_prompt=True
)
# 输出：<s>[INST] <<SYS>>\n你是一个AI专家助手<</SYS>>\n什么是人工智能？ [/INST]
```

#### Llama-3-Instruct

```python
messages = [
    {"role": "system", "content": "你是一个有帮助的AI助手"},
    {"role": "user", "content": "请解释深度学习"},
    {"role": "assistant", "content": "深度学习是机器学习的一个子领域..."},
    {"role": "user", "content": "有哪些应用？"}
]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# 输出：<|start_header_id|>system<|end_header_id|>\n\n你是一个有帮助的AI助手<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n请解释深度学习<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n深度学习是机器学习的一个子领域...<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n有哪些应用？<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
```

#### Mistral-Instruct v0.x

```python
messages = [
    {"role": "user", "content": "写一首关于AI的诗"},
    {"role": "assistant", "content": "在数字的海洋中漫游..."},
    {"role": "user", "content": "再写一首关于机器学习的"}
]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# 输出：<s>[INST] 写一首关于AI的诗 [/INST] 在数字的海洋中漫游...</s><s>[INST] 再写一首关于机器学习的 [/INST]
```

#### Gemma / Gemma-2

```python
messages = [
    {"role": "user", "content": "Python中的列表和元组有什么区别？"},
    {"role": "assistant", "content": "列表是可变的，而元组是不可变的..."},
    {"role": "user", "content": "那字典呢？"}
]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# 输出：<bos><start_of_turn>user\nPython中的列表和元组有什么区别？<end_of_turn>\n<start_of_turn>model\n列表是可变的，而元组是不可变的...<end_of_turn>\n<start_of_turn>user\n那字典呢？<end_of_turn>\n<start_of_turn>model
```

#### Yi-Chat

```python
messages = [
    {"role": "system", "content": "你是一个编程助手"},
    {"role": "user", "content": "如何用Python读取文件？"},
    {"role": "assistant", "content": "可以使用open函数..."},
    {"role": "user", "content": "能举个例子吗？"}
]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# 输出与Qwen相同格式
```

#### DeepSeek-Chat

```python
messages = [
    {"role": "system", "content": "你是一个DeepSeek AI助手"},
    {"role": "user", "content": "什么是强化学习？"},
    {"role": "assistant", "content": "强化学习是机器学习的一种..."},
    {"role": "user", "content": "它和监督学习有什么区别？"}
]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# 使用Qwen格式
```

#### GLM-4-Chat

```python
messages = [
    {"role": "system", "content": "你是一个GLM助手"},
    {"role": "user", "content": "介绍一下自然语言处理"},
    {"role": "assistant", "content": "自然语言处理是AI的重要领域..."},
    {"role": "user", "content": "有哪些应用场景？"}
]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# 输出：[gMASK]sop<|system|>\n你是一个GLM助手<|user|>\n介绍一下自然语言处理<|assistant|>\n自然语言处理是AI的重要领域...<|user|>\n有哪些应用场景？<|assistant|>
```

#### Baichuan2-Chat

```python
messages = [
    {"role": "system", "content": "你是一个百川助手"},
    {"role": "user", "content": "如何学习编程？"},
    {"role": "assistant", "content": "学习编程可以从基础开始..."},
    {"role": "user", "content": "推荐什么语言？"}
]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tokenizer_config={"add_bos_token": False}  # 重要：关闭BOS token
)
# 输出：<reserved_106>你是一个百川助手<reserved_107>如何学习编程？<reserved_108>学习编程可以从基础开始...<reserved_107>推荐什么语言？<reserved_108>
```

#### 通用使用技巧

```python
# 1. 检查模型是否支持chat_template
if tokenizer.chat_template is None:
    print("该tokenizer没有设置chat_template")

# 2. 查看模板内容
print(tokenizer.chat_template)

# 3. 强制使用特定模板（如果需要）
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model-name")
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    chat_template=None  # 或指定自定义模板
)
```

## Gemma

### 微调

#### LoRa

```python
# LoRA 比其它方案效果好的原因可能是因为它不增加深度，因为增加深度会导致训练不稳定。
# 另外，有人通过理论分析证明了只要秩（R）足够大，LoRA 就能适用于任意全连接神经网络。

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq,
    TrainingArguments, Trainer
)
from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM

# --- 1. 配置常量 ---
MODEL_PATH = r"/opt/model"
# 训练数据路径（JSON格式指令集）
DATA_PATH = r"./data/data.json"
# 输出路径
OUTPUT_DIR = r"/opt/model/gemma2-LoRa-output"

# Llama模板语法
PROMPT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "你是一个AI助手。<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}{input}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


# --- 2. 数据处理函数 ---
def process_func(example):
    instruction = PROMPT_TEMPLATE.format(instruction=example['instruction'], input=example.get('input', ''))
    response = f"{example['output']}<|eot_id|>"

    # add_special_tokens=False，因为你的模板里已经手动加了 <|begin_of_text|> 等
    ins_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    res_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    # 此时 input_ids 的长度是变长的
    input_ids = ins_ids + res_ids

    # 指令部分 (ins_ids) 设为 -100，表示计算 Loss 时忽略
    # 回答部分 (res_ids) 设为原 ID，表示需要学习
    labels = [-100] * len(ins_ids) + res_ids

    # 因为这里没有 Pad，所以全是有效内容，全是 1
    attention_mask = [1] * len(input_ids)

    # 在 process_func 返回前加一行防御性截断（只切不补）
    if len(input_ids) > 2048:  # 或者 2048，根据你显存决定
        input_ids = input_ids[:1024]
        attention_mask = attention_mask[:1024]
        labels = labels[:1024]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# --- 3. 主流程 ---
if __name__ == "__main__":
    # 3.1 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3.2 准备数据
    ds = Dataset.from_pandas(pd.read_json(DATA_PATH))
    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
    # 这是一个Dataset类型数据，里面包含input_ids、attention_mask、labels三个字段
        # attention_mask:告诉模型哪些 token 是有效的
        # labels:训练时的“正确答案”

    # 3.3 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"cuda":0},
        torch_dtype=torch.bfloat16,  # 推荐 bfloat16
        trust_remote_code=True
    )
    model.enable_input_require_grads()  # 开启梯度检查点兼容

    # 3.4 配置 LoRA (自动冻结非LoRA参数)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    # 拼装模型
    model = get_peft_model(model, peft_config)

    # 3.5 训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, # 训练批次大小，根据显存大小调整，大调大
        gradient_accumulation_steps=8, # 梯度累积，每4个step更新一次参数
        logging_steps=20,
        num_train_epochs=3,
        save_total_limit=2,
        learning_rate=1e-4,
        bf16=True,  # 开启半精度
        gradient_checkpointing=True, # 启用梯度检查点，节省资源
    )

    # 3.6 开始训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds, # 这个是个字典包列表
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=1024), # 字典内列表转张量
    )
    trainer.train()

    # 3.7 保存适配器 (Adapter)
    trainer.save_model(OUTPUT_DIR + "/final_adapter")
    print(f"LoRA训练完成，适配器已保存至: {OUTPUT_DIR}/final_adapter")

    # --- 4. (可选) 合并模型 ---
    # 注意：如果显存不足，请单独运行此步骤，不要在训练后紧接着运行
    # LoRA/AdaLoRA 这类适配器文件夹里必须带一个 adapter_config.json，这个文件中指明了基础模型位置
    # 这里无需再次指明基础模型位置，隐式的就加载进来了
    model_to_merge = AutoPeftModelForCausalLM.from_pretrained(
        OUTPUT_DIR + "/final_adapter",
        device_map={"cuda": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    merged_model = model_to_merge.merge_and_unload() # 合并模型
    merged_model.save_pretrained(OUTPUT_DIR + "/merged_model", safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR + "/merged_model")
```

#### PromptTuning

```python
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import pandas as pd
from peft import (
    PromptTuningConfig, get_peft_model, TaskType
)

model_path = r"/opt/model"
data_path = r"./data/data.json"
output_path = r"/opt/model/gemma2-PromptTuning-output"

# Gemma模板语法
prompt_template = (
    "<start_of_turn>system\n"
    "你是一个AI助手<end_of_turn>\n"
    "<start_of_turn>user\n"
    "{instruction}{input}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

def proc_func(example):
    """输出处理"""
    t = prompt_template.format(instruction=example["instruction"], input=example.get("input", ""))
    response = f"{example['output']}<end_of_turn>"

    inp_ids = tokenizer(t, add_special_tokens=False)["input_ids"]
    res_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    input_ids = inp_ids + res_ids
    attention_mask = [1] * len(input_ids)
    lab = [-100] * len(inp_ids) + res_ids

    # 判断过大值
    if len(input_ids) > 2048:
        input_ids = input_ids[:1024]
        attention_mask = attention_mask[:1024]
        lab = lab[:1024]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": lab
    }

# 加载数据
data = Dataset.from_pandas(pd.read_json(data_path))
# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# 调用map函数进行处理数据
data_to = data.map(proc_func, remove_columns=data.column_names)
# 创建模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"cuda":0},
    torch_dtype=torch.bfloat16
)
model.enable_input_require_grades()
# 创建微调配置
confi = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init_text="你是一个旅游助手",
    num_virtual_tokens=len(tokenizer("你是一个旅游助手")["input_ids"]),
    tokenizer_name_or_path=tokenizer
)
# 微调模型封装
model = get_peft_model(model, confi)
# 创建训练参数
args = TrainingArguments(
    output_dir = output_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    logging_steps=20,
    save_total_limit=3,
    learning_rate = 1e-4,
    bf16=True,
    num_train_epochs=10,
)
# 创建训练器
trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=data_to,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=1024),
)
# 训练
trainer.train()
# 保存适配器
# 这会在 output_path 下生成 adapter_model.safetensors 和 adapter_config.json
trainer.save_model(output_path)
tokenizer.save_pretrained(output_path)
# Prompt不支持基座与微调合并，你需要再使用时进行再加载
```

####  P-Tuning

```python
import torch
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import TaskType, get_peft_model, PromptEncoderConfig

model_path = r"/opt/model"
data_path = r"./data/data.json"
output_path = r"/opt/model/gemma2-PTuning-output"

# 模板语法
prompt_template = (
    "<start_of_turn>system\n"
    "你是一个AI助手<end_of_turn>\n"
    "<start_of_turn>user\n"
    "{instruction}{input}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

def proc_func(example):
    input = prompt_template.format(example["instruction"], example.get("input", ""))
    response = f"{example['output']}<end_of_turn>"

    ins_ids = tokenizer(input, add_specilal_tokens=False)["input_ids"]
    res_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    input_ids = ins_ids + res_ids
    attention_mask = [1] * input_ids
    label = [-100] * len(ins_ids) + res_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": label
    }

# 加载数据
data = Dataset.from_pandas(pd.read_json(data_path))
# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# 数据处理
data_handel = data.map(proc_func, remove_columns=data.column_names)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"cuda":0},
    torch_dtype=torch.bfloat16,
)
model.enable_input_require_grads()
# 加载微调配置（使用MLP默认是v2）
config = PromptEncoderConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=10,
    encoder_dropout=0.1,
    encoder_num_layers=5,
    encoder_hidden_size=1024
)
# 合并模型
model = get_peft_model(model, config)
# 训练配置
args = TrainingArguments(
    output_dir = output_path,
    per_device_train_batch_size=1,
    logging_steps=20,
    save_steps=20,
    save_total_limit=2,
    num_train_epochs=3,
    learning_rate=1e-4,
    bf16=True,
    gradient_checkpointing=True,
)
# 训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=data_handel,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=1024),
)
# 保存
trainer.save_model(output_path)
tokenizer.save_pretrained()
```

#### PrefixTuning

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import get_peft_model, TaskType, PrefixTuningConfig
import pandas as pd

model_path = r"/opt/model"
data_path = r"./data/data.json"
output_path = r"/opt/model/gemma2-PrefixTuning-output"

prompt_template = (
    "<start_of_turn>system\n"
    "你是一个AI助手<end_of_turn>\n"
    "<start_of_turn>user\n"
    "{instruction}{input}<end_of_turn>"
    "<start_of_turn>model\n"
)

# 处理函数
def proc_func(example):
    input = prompt_template.format(example["instruction"], example.get("input",""))
    response = f"{example['output']}"

    inp_ids = tokenizer(input, add_special_tokens=False)["input_ids"]
    res_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    input_ids = inp_ids + res_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(inp_ids) + res_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 加载数据
data = Dataset.from_pandas(pd.read_json(data_path))
# 初始化Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# 处理数据
data_handle = data.map(proc_func,  remove_columns=data.column_names)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"cuda":0},
    torch_dtype=torch.bfloat16
)
model.enable_input_require_grads()
# 微调配置
config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=10,
    prefix_projection=True
    # 优势：
        # 可能提升模型性能
        # 允许更灵活的前缀表示学习
    # 注意事项：
        # 增加少量额外参数
        # 训练时间可能略微增加
)
# 合并模型+微调配置
model = get_peft_model(model, config)
# 配置微调参数
arg = TrainingArguments(
    output_dir = output_path,
    per_device_train_batch_size=1,
    logging_steps=20,
    save_steps=20,
    save_total_limit=2,
    num_train_epochs=3,
    learning_rate=1e-4,
    bf16=True,
    gradient_checkpointing=True,
)
# 创建微调对象
trainer = Trainer(
    model=model,
    args=arg,
    train_dataset=data_handle,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=1024),
)
# 开始微调
trainer.train()
# 保存
trainer.save_model(output_path)
tokenizer.save_pretrained(output_path)
```

### 预测

#### Lora

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = r"/opt/model/merged_model"

# Llama模板语法
PROMPT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "你是一个AI助手。<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}{input}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


def load_model():
    """
    加载模型
    :return: tokenizer分词器对象, model模型对象
    """
    print("🔄 正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 这一步与训练时保持一致，加载基座
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"cuda":0},
        torch_dtype=torch.bfloat16,  # 保持和训练时一致的精度
        trust_remote_code=True
    )
    model.eval()  # 切换到评估模式
    return tokenizer, model


def gen_resp(tokenizer, model, query, input_text=""):
    """
    生成响应
    传入：tokenizer分词器对象, model模型对象, RAG_prompt数据库检索+用户输入的str数据
    :return: response模型响应的str数据
    """
    # 1. 格式化输入
    prompt = PROMPT_TEMPLATE.format(instruction=query, input=input_text)

    # 2. 编码
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 3. 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # 最大生成长度
            do_sample=True,  # 开启采样，增加多样性
            temperature=0.7,  # 温度控制
            top_p=0.9,  # 核采样
            repetition_penalty=1.1,  # 重复惩罚
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # 4. 解码 (只截取生成部分，去掉 Prompt)
    # 获取输入解析后的input_ids数量
    input_len = inputs.input_ids.shape[1]
    # 查询 + 答案，我们需要去掉查询，拿到生成的答案
    generated_ids = outputs[0][input_len:]
    # 解码答案，并且忽略特殊符号
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response


if __name__ == "__main__":
    # 加载模型
    tokenizer, model = load_model()

    # 待验证的虚构旅游地列表
    test_queries = [
        "介绍一下北京地摊文化博物馆。",
        "请问新疆海洋文化博物馆在哪里？有什么特色？",
        "给我讲讲北京地摊文化博物馆的镇馆之宝是什么？",
        "我想去新疆海洋文化博物馆旅游，有什么建议吗？"
    ]

    print("🚀 开始验证模型对虚构数据的掌握能力")

    for query in test_queries:
        print(f"👤 User: {query}")
        response = gen_resp(tokenizer, model, query)
        print(f"🤖 Assistant: {response}")
        print("-" * 50)
```

#### PromptTuning

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 路径
BASE_MODEL_PATH = r"/opt/model"
ADAPTER_PATH = r"/opt/model/gemma2-PromptTuning-output"

# Gemma
prompt_template = (
    "<start_of_turn>system\n"
    "你是一个AI助手<end_of_turn>\n"
    "<start_of_turn>user\n"
    "{instruction}{input}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map={"cuda":0},
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    tokenize = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model.eval()
    return model, tokenize

def gen_resp(tokenizer, model, query, input_text=""):
    """
        生成响应
        传入：tokenizer分词器对象, model模型对象, RAG_prompt数据库检索+用户输入的str数据
        :return: response模型响应的str数据
        """
    # 1. 格式化输入
    prompt = prompt_template.format(instruction=query, input=input_text)

    # 2. 编码
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 3. 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # 最大生成长度
            do_sample=True,  # 开启采样，增加多样性
            temperature=0.7,  # 温度控制
            top_p=0.9,  # 核采样
            repetition_penalty=1.1,  # 重复惩罚
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # 4. 解码 (只截取生成部分，去掉 Prompt)
    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response

if __name__ == "__main__":
    model, tokenizer = load_model()
    query = "北京有什么好玩的么？"
    print(gen_resp(tokenizer, model, query))
```

#### P-Tuning

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 路径
BASE_MODEL_PATH = r"/opt/model"
ADAPTER_PATH = r"/opt/model/gemma2-PTuning-output"

# 提示词模板
prompt_template = (
    "<start_of_turn>system\n"
    "你是一个AI助手<end_of_turn>\n"
    "<start_of_turn>user\n"
    "{instruction}{input}<end_of_turn>\n"
    "<start_of_turn>model\n"
)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map={"cuda":0},
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    return model, tokenizer

def gen_resp(tokenizer, model, query, input=""):
    prompt = prompt_template.format(instruction=query, input=input)
    input_ids = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output = model.generate(
            **input_ids,
            max_new_tokens=1024,
            do_sample=True
        )
    input_len = input_ids.input_ids.shape[1]
    gener_ids = output[0][input_len:]
    response = tokenizer.decode(gener_ids, skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model, tokenizer = load_model()
    query = "北京有什么好玩的么？"
    print(gen_resp(tokenizer, model, query))
```

#### PrefixTuning

```python
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL_PATH = r"/opt/model"
ADAPTER_PATH = r"/opt/model/gemma2-PrefixTuning-output"

prompt_template = (
    "<start_of_turn>system\n"
    "你是一个AI助手<end_of_turn>\n"
    "<start_of_turn>user\n"
    "{instruction}{input}<end_of_turn>"
    "<start_of_turn>model\n"
)

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"cuda":0}
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    return model, tokenizer

def gen_resp(tokenizer, model, query, input=""):
    prompt = prompt_template.format(
        instruction=query,
        input=input
    )
    input_ids = tokenizer(
        prompt,
        return_tensors='pt'
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **input_ids,
            max_new_tokens=1024,
            do_sample=True
        )
    input_len = input_ids.input_ids.shape[1] # 拿到input_ids第二维个数
    gener_ids = output[0][input_len:]
    response = tokenizer.decode(gener_ids, skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model, tokenizer = load_model()
    query = "北京有什么好玩的么？"
    print(gen_resp(tokenizer, model, query))
```