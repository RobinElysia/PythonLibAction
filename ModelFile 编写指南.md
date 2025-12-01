### Ollama Modelfile 权威指南

> 本指南面向想要在 Ollama 平台上打包、部署与调试模型的开发者与运维人员。涵盖 Modelfile 语法、常用字段、进阶技巧、示例与常见问题排查。内容以实践为导向，兼顾安全与可维护性。

#### 1. 简介：什么是 Modelfile

Modelfile 是用来告诉 Ollama 如何从基础镜像或模型文件构建运行时镜像和服务的声明性配置文件。它类似于 Dockerfile，但专注于 AI 模型的运行参数、模板与推理相关的配置。通过规范化的字段，开发者能指定模型来源、输入输出模板、运行时参数和资源要求。

#### 2. Modelfile 的基本结构

一个常见的 Modelfile 通常包含：

- 基础镜像或模型来源（FROM）
- 模板（TEMPLATE）用于定义系统/用户/assistant 对话上下文的基础结构
- PARAMETER：用于定义可调运行参数（如 temperature、max_tokens、stop 等）
- 可选：COPY/INSTALL 步骤（具体实现依平台支持）

基本示例：

```bash
FROM /opt/model/my-base-model
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|> # system部分结束
{{ range .Messages }}
<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }} # user and model 结束
<|im_start|>assistant # model开始
"""
PARAMETER temperature 0.7
PARAMETER max_tokens 1024
PARAMETER stop "<|im_end|>"
```

> 说明：不同 Ollama 版本对 Modelfile 的支持字段可能略有差异，建议在部署前参照你的 Ollama 服务版本文档。

#### 3. 常用字段详解

##### FROM

- 指定基础模型路径或镜像，例如一个本地模型目录或系统提供的基础镜像。
- 可以是相对路径或绝对路径，视平台而定。

##### TEMPLATE

- 用于定义对话格式（prompt template）。通常用多行文本包裹，支持占位符如 `{{ .System }}`, `{{ range .Messages }}` 等。
- 模板决定模型输入怎样被序列化到模型上下文，关系到 token 使用和输出一致性。

##### PARAMETER

- 用于定义运行时参数。每一行设置一个键值对。
- 常见参数：`temperature`, `top_p`, `max_tokens`, `stop`, `num_beams`, `repetition_penalty` 等。
- 值可以是数字、字符串或布尔值，取决于参数类型。

##### STOP

- 有些实现把 stop 作为专门字段。它用于定义输出终止符号或 token 序列，避免生成过长或滥生成。

#### 4. 模板（TEMPLATE）与停止符（STOP）

- TEMPLATE 的设计原则：**简洁**、**确定性**、**低 token 消耗**。
- 使用系统（system）角色放置指令与安全限制，用户（user）角色放置真实输入。
- 在需要保证回答不带额外说明或结束在指定 token 时，配置合适的 `stop` 参数。

示例模板（指令风格）：

```bash
TEMPLATE """
<|im_start|>system
You are a helpful assistant. Follow instructions precisely.
<|im_end|>
<|im_start|>user
{{ .Content }}
<|im_end|>
<|im_start|>assistant
"""
```

```bash
<|im_start|>system
You are a helpful assistant. Follow instructions precisely.
<|im_end|>
# 定义了系统角色内容（system message）。
# 告诉模型它的身份和行为规范。
# 这部分通常用于放安全策略、人格设定、任务边界等。
# 这段相当于：“你是一个乐于助人的助手，严格按指令办事。”

<|im_start|>user # 这里制定了必须是user，也可以是Role
{{ .Content }}
<|im_end|>
# 这一段表示用户输入的内容。
# {{ .Content }} 是一个变量，会被 Ollama 在运行时替换为实际的用户问题，例如：“请帮我规划一次日本5日游”。

<|im_start|>assistant
# 这里没有 <|im_end|>，表示模型的回答应接在这里生成。  
# 模型看到这个提示时，知道该轮到“assistant”说话了。
# 它会从这里开始生成文字，直到遇到指定的停止标志（stop token，比如 <|im_end|>）。
```

停止符示例：`PARAMETER stop "<|im_end|>"`，当模型生成完成后，是这个样子的：

```bash

<|im_start|>assistant
我推荐您考虑日本北海道或冰岛，这些地方在夏季凉爽宜人，风景优美。
PARAMETER stop "<|im_end|>"

```

总结：

```bash

TEMPLATE """<|im_start|>system
{{ .System }}              # 系统提示
<|im_end|>
{{ range .Messages }}       # 遍历历史消息
<|im_start|>{{ .Role }}     # 当前消息角色（user/assistant/system）
{{ .Content }}               # 当前消息文本
<|im_end|>
{{ end }} # {{ range .Messages }} 是一对的
<|im_start|>assistant        # 模型输出起始点
"""

```

#### 5. 参数（PARAMETER）详解与推荐值

- `temperature`（通常 0.0 - 1.0）: 控制随机性；任务需要确定性时设为 0；需要创造性时设高（0.7-1.0）。
- `top_p`（0.0 - 1.0）: 过滤候选 token 的累积概率阈值；与 temperature 联合使用可更好控制输出。
- `max_tokens`：回答最大 token 数；根据上下文与成本设定（例如 512，1024，2048）。
- `stop`：停止序列，字符串或数组形式。
- `num_beams`：用于束搜索，增加确定性但更慢（通常 1-5）。
- `repetition_penalty`：惩罚重复，避免循环生成。

示例：

```bash
PARAMETER temperature 0.0
PARAMETER top_p 0.9
PARAMETER max_tokens 512
PARAMETER stop "<|im_end|>"
```

#### 6. 常见使用场景示例

##### 6.1 简单推理服务（聊天机器人）

```bash
FROM /opt/model/ggml-model
TEMPLATE """<|im_start|>system
You are a helpful assistant.
<|im_end|>
{{ range .Messages }}
<|im_start|>{{ .Role }}
{{ .Content }}
<|im_end|>
{{ end }}
<|im_start|>assistant
"""
PARAMETER temperature 0.2
PARAMETER max_tokens 512
PARAMETER stop "<|im_end|>"
```

##### 6.2 指令微调（SFT）后的模型部署

对 SFT 模型，建议：
- 在 TEMPLATE 中明确 system 指令，避免模型偏离调优目标。
- 将 SFT 权重放在安全路径并从 `FROM` 指向。

```bash
FROM /mnt/models/sft-model
TEMPLATE """
<|im_start|>system
You are an assistant specialized in legal summarization. Keep answers concise.
<|im_end|>
"""
PARAMETER temperature 0.0
PARAMETER max_tokens 400
```

##### 6.3 大模型量化与低内存部署

- 使用量化（8-bit、4-bit）以降低显存需求。
- 在 Modelfile 上标注模型已量化且只适用特定后端（例如 GGML、bitsandbytes）。

并非强制要求，但是是约定

```bash
FROM /opt/model/llama-7b-q4
PARAMETER quantized true # 标注是否量化
PARAMETER backend ggml
# 这个模型文件是 GGML/GGUF 格式，只能用 CPU 或 llama.cpp 生态的运行时加载。
# 还有：
1. gguf：新一代 llama.cpp 格式，ggml 的升级版
2. bitsandbytes：GPU 后端，支持 8-bit/4-bit 权重
3. cuda：显式声明使用 NVIDIA CUDA GPU 后端
4. metal：Mac 平台的 GPU 加速后端
5. cpu：强制使用 CPU 推理
6. vulkan：部分系统的跨平台 GPU 后端
7. tensorrt：NVIDIA 高性能推理引擎
```

> 提示：量化后测试模型输出一致性，必要时在参数中适当增加 `repetition_penalty`。

#### 7. 附录：示例 Modelfile 热门变体

##### 最小可用 Modelfile

```bash
FROM /opt/model/small-ggml
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
{{ range .Messages }}
<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}
<|im_start|>assistant
"""
PARAMETER temperature 0.5
PARAMETER max_tokens 256
PARAMETER stop "<|im_end|>"
```

##### 完整的 Demo

```bash
FROM /opt/model/qwen2.5-1.8b-instruct
TEMPLATE """
<|im_start|>system
You are TourTools LP, a professional travel planning assistant.
Your goals:
1. Always collect the following user information before giving travel suggestions:
   - gender
   - age
   - travel season
   - travel duration
   - budget
2. Politely refuse unrelated questions and redirect the user back to travel planning topics.
3. 你仅回答旅游相关的内容，其余的一概回复抱歉，我无法完成
4. Always end your response with:
   '感谢您的惠顾，依据您的需求，为您提供如下建议：'

<|im_end|>
{{ range .Messages }}
<|im_start|>{{ .Role }}
{{ .Content }}
<|im_end|>
{{ end }}
<|im_start|>assistant
"""
PARAMETER temperature 0.3
PARAMETER stop "<|im_end|>"
```