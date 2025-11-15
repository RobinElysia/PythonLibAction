# PythonLibAction！
💡 探索、学习、贡献！这是一个开放的 Python AI 第三方库学习社区。我们不仅收集和解析最酷的 Python AI 库，还鼓励你通过提交代码和案例来共同构建这份知识图谱。一起点亮 Python AI 技能树吧！

## 环境配置
准备工作：
1. 虚拟机：WSL Ubuntu 20.04 以上版本（不含20.04，低版本系统处理方案我会放在文中）
2. Python：3.10
3. [virtualenv：任意版本](https://pypi.org/project/virtualenv/#files)
4. [CUDA：11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0)
5. [cuDNN：CUDA 11.X → cuDNN 8.9.7](https://developer.nvidia.com/rdp/cudnn-archive)
6. [PyTorch：2.0.1](https://pypi.org/project/torch/2.0.1/#files)
7. [Tensorflow：2.12.1](https://pypi.org/project/tensorflow/2.12.1/#files)

注意：Python、CUDA、cuDNN、PyTorch、Tensorflow这五者拥有较强的依赖关系，其中 CUDA and cuDNN 是强依赖关系，CUDA 与 Pytorch、Tensorflow 是强依赖关系。因为我这里使用的是离线安装，很大程度上需要自行处理依赖关系，所以特此强调。对于直接 pip install XXX 的在线用户，大概率无需考虑 pytorch 和 tensorflow 的版本问题。

### 虚拟环境与 Python 安装

这里采用 WSL2 作为本地开发环境支持，为什么不选 VM 和 其他的虚拟化技术，一是 VM 有点大材小用的感觉，虚拟化环境配发 4 核心或许不太够用；为什么不用 Docker 虚拟环境，一方面是出现的各种硬件虚拟化问题，一方面是精简虚拟化环境可能导致各种意料之外的问题。所以我们采用折中方案， WSL2 虚拟化。

对于 WSL2 而言，我建议使用 Ubuntu 22 以上的版本，Python 环境无需有过多的担忧，20 及以下版本 apt 无 py3.8 以上版本，需要手动编译高版本 py，我会给出详细的高低版本教程。

#### 高版本系统

```bash
# 更新
sudo apt upgrade
sudo apt update

# 查看是否存在高版本 py
sudo apt list python3
# 查看是否存在 3.9 以上版本

# 选择存在的高版本进行安装
sudo apt install python3.9/python3.10/...

# 验证
python3 --version

# 出现正常回显代表成功
Python 3.10.17
```

#### 低版本系统

低版本较为麻烦，需要去 python 官网或者ustc源直接下载高版本的压缩包，解压缩编译

这里我们选择 ustc源 进行下载，链接如下：
https://mirrors.ustc.edu.cn/python/3.10.8/Python-3.10.8.tar.xz

步骤如下：

```bash

# 进入到opt中
cd /opt
# 下载安装包
wget https://mirrors.ustc.edu.cn/python/3.10.8/Python-3.10.8.tar.xz
# 解压缩
tar -xvf Python-3.10.8.tar.xz
# 进入文件夹
cd Python-3.10.8

# 更新，安装gcc/g++/make
sudo apt update
sudo apt install -y build-essential

# 安装 zlib1g 依赖
sudo apt install -y zlib1g-dev libssl-dev libbz2-dev libreadline-dev \
  libsqlite3-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev

# 验证
gcc --version   # 应能看到版本号

# 在 Python-3.10.8 中进行编译
sudo ./configure --prefix=/usr/local/python3.11 \
                 --enable-optimizations \
                 --with-ensurepip=install
                 

sudo make -j$(nproc)
sudo make install

# 编译完成就成功了

```

比赛的时候，这一步 py 环境的搭建是没有的，毕竟连最基本的环境都没有，还打什么比赛？？？

### CUDA 与 cuDNN

#### CUDA

```bash

# 在普通用户下，检测最高支持的 CUDA 版本
nvidia-smi # 这一步比赛忽略

# 直接安装比最高支持版本低的版本
cd /opt && wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run # 这一步可忽略，因为比赛方会提供文件

# 运行
sudo su cuda_11.8.0_520.61.05_linux.run
# 时间会很长，因为文件很大
后边就是 accept 即可

# 环境配置
vim ~/.bashrc
export PATH=/usr/local/cuda-11.8/bin=${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64=${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/suda-11.8
# 保存退出
# 刷新
source ~/.bashrc

# 验证
nvcc -V
# 回显
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0

```

#### cuDNN
压缩包安装：

```bash
# 下载
wget https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz/ # 比赛忽略这一步

# 解压缩
tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz
# 进入解压后的文件夹
cd cudnn-linux-x86_64-8.9.7.29_cuda11-archive
# 移动文件到原来 CUDA 安装的文件夹下
sudo cp include/* /usr/local/cuda-11.8/include
sudo cp lib/libcudnn* /usr/local/cuda-11.8/lib64
# 你需要确认以下文件夹名对不对

# 给予权限
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# 安装完成
```

deb安装：
```bash

# dpkg
dpkg -i XXX.deb

```

### 安装 Virtualenv、Pytorch、Tensorflow、Ollama、Model

#### Virtualenv

```bash

# 进入 opt
cd /opt
# 下载 whl
wget https://files.pythonhosted.org/packages/79/0c/c05523fa3181fdf0c9c52a6ba91a23fbf3246cc095f26f6516f9c60e6771/virtualenv-20.35.4-py3-none-any.whl # 比赛忽略

# 安装
pip install /opt/virtualenv-20.35.4-py3-none-any.whl

```

以下是直接在全局进行安装 Pytorch Tensorflow
#### Pytorch

```bash

# 在 opt 下
wget https://files.pythonhosted.org/packages/21/33/4925decd863ce88ed9190a4bd872b01c146243ee68db08c72923984fe335/torch-2.0.1-cp310-cp310-manylinux2014_aarch64.whl

# 安装
pip install /opt/torch-2.0.1-cp310-cp310-manylinux2014_aarch64.whl

```

#### Tensorflow

```bash

# 在 opt 下
wget https://files.pythonhosted.org/packages/21/33/4925decd863ce88ed9190a4bd872b01c146243ee68db08c72923984fe335/torch-2.0.1-cp310-cp310-manylinux2014_aarch64.whl

# 安装
pip install /opt/torch-2.0.1-cp310-cp310-manylinux2014_aarch64.whl

```

或者使用Virtualenv新建虚拟环境进行安装：
#### Virtualenv 虚拟环境下安装环境（不如 uv

```bash

# 假如在 /home/user/test 下
virtualenv -p python3.10 ProjectName
# 新建虚拟环境 ProjectName
# 激活虚拟环境
source /home/user/test/ProjectName/bin/activate

# 安装依赖
pip install /opt/torch-2.0.1-cp310-cp310-manylinux2014_aarch64.whl && pip install /opt/torch-2.0.1-cp310-cp310-manylinux2014_aarch64.whl

# 验证 CUDA、cuDNN、Pytorch、Tensorflow是否安装成功
# Pytorch
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.backends.cudnn.version())"
# Tensorflow
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

#### Ollama、模型安装
```bash
# 解压Ollama
sudo tar -C /usr/ollama -xvf 文件名.文件后缀
# 环境
vim /etc/profile
export OLLAMA_HOME=/usr/ollama
export PATH=$OLLAMA_HOME/bin:$PATH

# 创建 Modelfile
vim Modelfile
# 注意，不同用户需要都刷新

# 文件中引入模型
# 创建新的 Modelfile 文件
# 怎么编写参考下述文档
# 保存退出

# 创建模型
ollama create 好记的名称 -f Modelfile

# 运行模型
ollama run 好记的名称

# 你就可以愉快的 ask 了
```

关于`Ollama Modelfile`和部分Modelfile中的系统调试，你可以参考[下文](#Ollama Modelfile 权威指南)

### OpenWebUI 安装

注：关于open-webui，你需要开启wsl网络镜像，同时开启vpn加速，关闭全局代理。

```bash
# 进入opt下
cd /opt/openwebui

# 下载
wget https://files.pythonhosted.org/packages/a6/dd/3665ce90ca299d670c656d767effb33c6a20b6370361e7f11cd1f72e947b/open_webui-0.6.36-py3-none-any.whl

# 安装
pip install /opt/openwebui/open_webui-0.6.36-py3-none-any.whl

# 安装pysqlite3
pip install pysqlite3-binary # 可忽略

# 修改对应的init文件
...# 可忽略

# 运行
open-webui serve

```

### 其他
标准库：json、os、xml和re无需下载，直接导入使用
第三方库：
1. pymysql：直接pip install就可以
2. pymongo：同上
3. faiss：Windows平台仅支持cpu版本，py版本在3.14~9（我使用的是3.10.X版本）；gpu加速版本仅限Linux平台，版本限制在3.10~6。官方建议在conda虚拟环境上安装，但是没必要，我将使用权威的 uv。
4. sqlite3：仅限Linux平台，自己搭建可能需要手动编译>=3.35.0版本，编译完成后重新让Py链接sqlite3的依赖（这部分如果你的py也是编译的，那就重新编译一遍）

## 慢速开始
注： 相关 Py 源代码见个人仓库
### Linux 使用技巧
#### Vim
在默认模式下 `gg` 跳转到文件开头；`G` 跳转到末尾；`数字 + G` 跳转到指定行数
默认模式下 `H` 跳转到当前窗口首行；`M` 跳转到中间；`L` 最后行
默认模式下输入 `/关键字` 可以进行搜索；`n` 下一个；`N` 上一个结果
默认模式下输入 `?关键字` 可以进行向上搜索
默认模式下输入 `:changes` 查看文件编辑历史

buffer
默认模式下输入 `:e 文件`，`:ls` 查看所有打开的`buffer`文件，我们可以使用 `:b 编号` 实现快速切换，或者使用`:bn`和`:bp`进行切换；关闭此`buffer`使用`bd`
windows
默认模式下输入 `:vs` 左右分屏、`:sp`上下分屏；多次按 `ctrl+w` 进行切换；
tabe
默认模式下输入 `:tabe 文件名` 创建一个新的页面，你可以打开多个文件。使用`先g后t`进行切换

### 正则表达式（讨厌re的原因是因为它是设计给天才看的）
```bash
# 单字符限定
? 代表前 1 个字符或 0 个字符
	abb?：ab或者abb

* 代表匹配 0 个、 1 个或 多个字符
	ac*b：acb、ab、accccb...
	
+ 代表匹配 1 个以上的字符个数
	ac+b：acb、accb、acccb...

限定出现次数：
	ac{2,6}b：accb、acccb、accccb、acccccb、accccccb # 2到6个c
	ac{2,}b：accb、acccb、accccb... # 2个以上

# 多字符限定
使用 (.) 括起来：

(ab)+b：abb、ababb、abababb... # ab整个字符重复出现多次，多字符限定
# 其他的限定符：？、* 同理

# 或运算
a (cat|dog)：a cat 或者 a dog
a cat|dog：a cat 或者 dog

# 字符类
限定仅用 指定字符 进行组成的字符串
[abc]+：abc、aabbcc、abccba
[a-zA-Z0-9]+：a到z、A到Z、0到9
[^0-9]+：除了0到9

# 元字符
\d+：数字
\w+：单词
\s：tap和换行
\D+：非数字
\W+：非单词
\S+：非空白字符
.*：除了换行其余的字符
^a：匹配行首的a
b$：匹配行尾b

# r'' r代表这是py得原始字符、与之对应的就是普通字符
```

#### 实践

```bash

# 匹配所有 HTML 标签
<.+?>

# RGB 匹配
#[0-9A-Fa-f]{6}\b
# 单词和空格间的位置。例如， 'er\b' 可以匹配"never" 中的 'er'，但不能匹配 "verb" 中的 'er'。

# IPV4 地址
(25[0-5]|2[0-4]\d|[01]?\d?\d\.){3}(25[0-5]|2[0-4]\d|[01]?\d?\d)\b
# (25[0-5]|2[0-4]\d|[01]?\d?\d\.){3} 代表匹配 A.B.C.
# (25[0-5]|2[0-4]\d|[01]?\d?\d) 代表匹配 D
# 加上 \b 后，相连其他不会被匹配，比如：192.168.0.1abc
```

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
{{ .System }}<|im_end|>
{{ range .Messages }}
<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}
<|im_start|>assistant
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

##### 样题解答

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

### 数据处理

#### JSON（万物基于JSON）
##### 代码
```python
import json as json  
  
  
# json转字典  
with open("test.json", "r", encoding="utf-8") as f:  
    data = json.load(f) # 读取 json 文件  
  
name = data["name"]  
hobbies: list = data["hobbies"]  
email = data["contact"]["email"]  
  
print(name)  
for i, hobby in enumerate(hobbies):  
    print(i, hobby)  
print(email)  
  
# 字典转json  
DataMap = json.dumps(data)  
print(DataMap)  
with open("OutTest.json", "w", encoding="utf-8") as f:  
    json.dump(  
        data, # 输入字典  
        f, # 输出文件  
        ensure_ascii=False, # 将所有非ASCII字符转换为Unicode转义序列  
        indent=4, # 缩进  
        sort_keys=False # 保持键的原始插入顺序，非排序  
    )
```

##### 测试 JSON 文件
```JSON
{  
  "name": "张三",  
  "age": 25,  
  "city": "北京",  
  "occupation": "软件工程师",  
  "hobbies": ["读书", "游泳", "编程"],  
  "contact": {  
    "email": "zhangsan@example.com",  
    "phone": "13800138000"  
  },  
  "is_student": false,  
  "score": 95.5  
}
```

#### XML（谁？不熟，感觉是Java佬最爱）
##### XML 数据清洗
```python
import xml.etree.ElementTree as et  
import xml.dom.minidom as Doc  
  
# 解析xml文件  
root = et.ElementTree(file="test.xml") # 创建ElementTree对象  
print("对象：",root)  
tree = root.getroot() # 获取根节点  
print("根节点对象：", tree)  
print("根节点标签：", tree.tag) # 获取根节点标签，就是最外层的标签  
print("根节点属性：", tree.attrib) # 获取根节点属性  
  
# 遍历获取子元素  
for child in tree:  
    # print("子元素标签：", child.tag,"子元素属性：", child.attrib)  
    print(child[0].text)  
    print(child[1].text)  
    print(child[2].text)  
    for grandchild in child:  
         print("孙元素标签：", grandchild.tag, "孙元素文本", grandchild.text)  
  
# 索引获取孙元素  
print(tree[0][0].text)  
  
# 查询元素  
print(tree.find("company")) # 查询子元素  
print(tree.find("person").find("name").text) # 查询首次出现的孙元素  
print(tree.findall("person")[1].find("name").text) # 查询第二个孙元素  
  
print("-----------------")  
# 创建xml文件  
doc = Doc.Document() # 创建Document对象  
root = doc.createElement("root") # 创建根节点  
doc.appendChild(root) # 添加根节点  
  
head = doc.createElement("head") # 创建子节点  
root.appendChild(head) # 添加子节点  
  
ch = doc.createElement("ch") # 创建孙节点  
head.appendChild(ch) # 添加孙节点  
  
text = doc.createTextNode("hello world")  
ch.appendChild(text) # 添加文本节点  
  
print(doc.toxml()) # 输出xml文件  
  
# 保存xml文件  
# with open("OutTest.xml", "w+") as f:  
#     f.write(  
#         doc.toprettyxml(encoding="utf-8")  
#         .decode("utf-8")  
#     )# 保存 xml 文件  
  
print("-----------------")  
par = et.parse("test.xml")  
print(par)  
# 剩下的和解析xml文件一样，par 是一个 ElementTree 对象
```

##### 测试 XML 文件
```xml
<?xml version="1.0" encoding="UTF-8"?>  
<root title="你好">  
    <person id="1">  
        <name>张三</name>  
        <age>25</age>  
        <email>zhangsan@example.com</email>  
    </person>
    <person id="2">  
        <name>李四</name>  
        <age>30</age>  
        <email>lisi@example.com</email>  
    </person>
    <company>
	    <name>示例科技有限公司</name>  
        <founded>2010</founded>  
        <employees>50</employees>  
    </company>
</root>
```

#### HTML（BeautifulSoup，漂亮的汤（
##### HTML 数据清洗
```python
from bs4 import BeautifulSoup  
  
with open("test.html", "r", encoding="utf-8") as f:  
    soup = BeautifulSoup(f, "lxml")  
"""  
使用BS创建对象  
参数一：可以是html的字符串，或者文件对象  
参数二：指定解析器，或者不指定使用默认  
"""  
print("________基本调用__________")  
print(soup.title) # 获取第一个title节点  
print(soup.meta) # 获取第一个meta节点  
  
print("________三个重要信息__________")  
  
# 获取标签的三个重要信息  
# 内容  
print(soup.a.string)  
# 属性  
print(soup.a.attrs)  
# tag名称  
print(soup.a.name)  
  
print("________子标签调用__________")  
# tag.tag  
# 内容  
print(soup.li.a.string)  
# 属性  
print(soup.li.a.attrs)  
# tag名称  
print(soup.li.a.name)  
  
print("________关联选择__________")  
# 子节点  
print(soup.li.contents) # 返回列表  
print(soup.li.children) # 返回迭代器  
for i in soup.head.children:  
    print("子节点", i)  
  
# 父节点  
print("父节点" , soup.meta.parent)  
  
# 祖先节点  
print(soup.a.parents) # 生成器  
print("祖先节点:" , list(enumerate(soup.ul.parents)))  
  
# 兄弟节点  
print("兄弟节点:" , soup.meta.next_sibling) # 下一个兄弟节点  
print("兄弟节点:" , list(  
    enumerate(soup.meta.next_siblings))) # 下一个兄弟节点生成器  
print("兄弟节点:" , soup.title.previous_sibling) # 上一个兄弟节点  
print("兄弟节点:" , list(  
    enumerate(soup.title.previous_siblings))) # 上一个兄弟节点生成器  
  
print("________css选择器__________")  
print(soup.select("section label")) # 标签选择器，html44、47行  
print(soup.select("#section3 .aaa")) # id + class选择器，41行  
print(soup.select("#section3 #name")) # 标签id选择器，45行  
  
print("________css选择器高级使用方法__________")  
for i in soup.select("form"):  
    print(i.select("input")[0].attrs) # input标签，html45行  
    print(i.select("label")[0].string) # 44行  
    # i.select("input")[0]、i.select("label")[0]是一个tag  
  
import re  
print("________方法选择器__________")  
print("获取所有a标签", soup.find_all(name="a"))  
print("获取id为name的标签", soup.find_all(id="name"))  
print("获取class为aaa的标签", soup.find_all(class_="aaa"))  
print("属性值查询，获取class为aaa的标签", soup.find_all(attrs={"class": "aaa"}))  
print("获取文本为 章节1 的标签", soup.find_all(string="章节1"))  
print("获取以章节开头的标签，正则匹配", soup.find_all(name=re.compile('^in')))  
print("获取以章节开头的标签，正则匹配，限制数量", soup.find_all(name=re.compile('^in'), limit=1))  
print("获取以章节为开头的", soup.find_all(string=re.compile("^章节")))  
print("获取所有标签", soup.find_all(True))  
  
# find 符合条件的第一个元素，使用上除了limit，其他与find_all一样  
print("find", soup.find(name="a"))  
print("find", soup.find(id="name"))  
print("find", soup.find(class_="aaa"))  
print("find", soup.find(attrs={"class": "aaa"}))  
print("find", soup.find(string="章节1"))
```

##### 测试 HTML 文件
```HTML
<!DOCTYPE html>  
<html lang="zh-CN">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>HTML5 测试文件</title>  
</head>  
<body>  
    <header>
	    <h1>欢迎来到HTML5测试页面</h1>  
    </header>  
    <nav>
		<ul>
			<li><a href="#section1">章节1</a></li>  
            <li><a href="#section2">章节2</a></li>  
            <li><a href="#section3">章节3</a></li>  
        </ul>
	</nav>  
    <main>
	    <section id="section1">  
            <h2>章节1</h2>  
            <p>这是HTML5测试文件的第一个章节。</p>  
            <article>
	            <h3>文章标题</h3>  
                <p>这是一篇示例文章的内容。</p>  
            </article>
	    </section>  
        <section id="section2">  
            <h2>章节2</h2>  
            <p>这里展示了一些HTML5的新元素。</p>  
            <aside>                <h4>侧边栏</h4>  
                <p>这是页面的附加信息。</p>  
            </aside>        </section>  
        <section id="section3">  
            <h3 class="aaa">HTML5新元素</h3>  
            <h2>章节3</h2>  
            <form>                <label for="name">姓名:</label>  
                <input type="text" id="name" name="name" required>  
  
                <label for="email">邮箱:</label>  
                <input type="email" id="email" name="email" required>  
  
                <button type="submit">提交</button>  
            </form>        </section>    </main>  
    <footer>        <p>&copy; 2023 HTML5测试文件</p>  
    </footer></body>  
</html>
```

#### pandas（你为什么不学熊猫？）
##### 样本数据生成
```python
import pandas as pd  
import numpy as np  
from datetime import datetime, timedelta  
import random  
  
# 设置随机种子以保证结果可重现  
np.random.seed(42)  
random.seed(42)  
  
# 生成100行数据  
n_rows = 100  
  
# 创建复杂的数据集  
data = {  
    # 1. 客户ID (字符串与数字混合)  
    '客户ID': [f'CUST{str(i).zfill(5)}' for i in range(1000, 1000 + n_rows)],  
  
    # 2. 交易日期 (时间序列)  
    '交易日期': [datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365)) for _ in range(n_rows)],  
  
    # 3. 产品类别 (分类变量)  
    '产品类别': random.choices(['Electronics', 'Clothing', 'Food', 'Books', 'Home'],  
                                       weights=[0.3, 0.25, 0.2, 0.15, 0.1], k=n_rows),  
  
    # 4. 销售额 (有异常值的连续变量)  
    '销售额': np.random.lognormal(mean=5, sigma=1.2, size=n_rows).round(2),  
  
    # 5. 客户评分 (1-5的离散评分，包含缺失值)  
    '客户评分': [random.randint(1, 5) if random.random() > 0.1 else np.nan for _ in range(n_rows)],  
  
    # 6. 地区 (地理分类)  
    '地区': random.choices(['North', 'South', 'East', 'West', 'Central'], k=n_rows),  
  
    # 7. 交易状态 (布尔与分类混合)  
    '交易状态': random.choices(['Completed', 'Failed', 'Pending', 'Refunded'],  
                                         weights=[0.7, 0.1, 0.15, 0.05], k=n_rows),  
  
    # 8. 利润 (与销售额相关但有噪声)  
    '利润': []  
}  
  
# 生成利润数据，与销售额相关但加入噪声  
for sales in data['销售额']:  
    base_profit = sales * random.uniform(0.1, 0.4)  # 利润率10%-40%  
    noise = base_profit * random.uniform(-0.2, 0.2)  # ±20%的噪声  
    data['利润'].append(round(base_profit + noise, 2))  
  
# 创建DataFrame  
df = pd.DataFrame(data)  
  
# 添加一些数据质量问题  
# 1. 在销售额中添加几个异常值  
outlier_indices = random.sample(range(n_rows), 5)  
for idx in outlier_indices:  
    df.loc[idx, '销售额'] *= 10  
  
# 2. 在地区中添加一些不一致的大小写  
mixed_case_indices = random.sample(range(n_rows), 8)  
for idx in mixed_case_indices:  
    df.loc[idx, '地区'] = df.loc[idx, '地区'].lower()  
  
# 保存为CSV文件  
df.to_json("complex_sales_data.json",  index=False)
# df.to_json("complex_sales_data_json.json", orient="records", indent=4, force_ascii=False)    生成JSON测试数据

# 保存为JSON文件,  orient="records" 表示将DataFrame转换为JSON列表, indent=4 表示缩进4个空格,  
# force_ascii=False 表示将所有非ASCII字符转换为Unicode转义序列  
  
print("CSV文件已生成: complex_sales_data.csv")  
print(f"文件形状: {df.shape}")  
print("\n数据前5行:")  
print(df.head())  
print("\n数据类型:")  
print(df.dtypes)  
print("\n基本统计信息:")  
print(df.describe())  
print("\n缺失值统计:")  
print(df.isnull().sum())
```

##### Series
```python
import pandas as pd  
import numpy as np  
  
# 创建单列Series数据  
s1 = pd.Series([1, 2, 3, 4, 5])  
s2 = pd.Series([1, 2, 3, 4, 5], dtype="float64", name="series_name", index=["a", "b", "c", "d", "e"])  
# 可自定义索引值、自定义类型、自定义列名  
print(s1)  
print(s2)  
  
# 字典创建  
s3 = pd.Series({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})  
print(s3)  
# 根据Series对象提取数据  
s4 = pd.Series(s3, index=["a", "c"])  
print(s4)  
print("_____Series属性_____")  
# 属性  
print(s2.index) # 索引  
print(s2.values) # 值  
print(s2.ndim) # 维度  
print(s2.shape) # 形状  
print(s2.size) # 大小  
print(s2.dtype) # 数据类型  
print(s2.name) # 列名  
  
print(s2.loc["a"]) # 自定义id索引获取值，显式索引  
print(s2.loc["a":"c"]) # 自定义id切片获取值，显式索引，左闭右闭  
print(s2.iloc[0]) # 默认索引获取值，隐式索引  
print(s2.iloc[0:3]) # 默认索引切片获取值，隐式索引，左闭右开  
  
print(s2.at["a"]) # 自定义id索引获取值，显式索引  
# 不支持切片  
print(s2.iat[0]) # 默认索引获取值，隐式索引  
# 不支持切片  
  
print("_____访问数据_____")  
# 标签取值  
print(s2["a"])  
# 布尔索引  
print(s2[s2 > 2])  
# 查看头5行信息（默认是前5行）  
print(s2.head(5)) # 我们只有5行  
# 查看尾5行信息（默认是后5行）  
print(s2.tail(5)) # 我们只有5行  
  
print("_____常用方法_____")  
s5 = pd.Series([1, 2, 3, np.nan, 4, 5, None], index=["a", "b", "c", "d", "e", "f", "g"]) # 创建Series对象  
print(s5) # np.nan和None统一处理为NaN  
print("前5行", s5.head()) # 默认是前5行  
print("后5行", s5.tail()) # 默认是后5行  
print("描述信息：", s5.describe())  
"""  
count: 去掉缺失值的个数  
mean: 平均值  
std: 标准差  
min: 最小值  
max: 最大值  
"""  
print("去掉缺失值个数：", s5.count())  
print("索引值", s5.keys()) # 等价于 s5.indexprint("索引值", s5.index)  
print("布尔是否为缺失值", s5.isna())  
print("布尔是否在列表中", s5.isin([1, 2, 3, 4, 5]))  
print("均值个数", s5.mean())  
print("最小值个数", s5.min())  
print("最大值个数", s5.max())  
print("标准差个数", s5.std())  
print("方差个数", s5.var())  
print("和个数", s5.sum())  
print("中位数", s5.median())  
print("百分位数", s5.quantile([0.5, 0.75]))  
print("25百分位数", s5.quantile(0.25))  
print("众数", s5.mode())  
print("值计数", s5.value_counts())  
print("去重", s5.unique()) # 把nan也去了，这个是列表  
# 或者  
print("去重", s5.drop_duplicates()) # 与unique输出的数据类型不同，这个是Series  
print("去重之后的个数", s5.nunique())  
# 排序  
print("以值排序", s5.sort_values())  
print("以值排序", s5.sort_values(ascending=False)) # 降序  
print("以索引排序", s5.sort_index())  
# 差值  
print("差值", s5.diff())  
print("差值绝对值", s5.diff().abs()) # 取绝对值  
  
# 时间  
s7 = pd.Series(np.random.randn(5), index=pd.date_range("2020-01-01", periods=12, freq="MS"))  
# 从 2020-01-01 00:00:00 开始，5个时间点，间隔为1个月  
s7 = s7.resample("QS").mean() # 获取QS频率的采样对象，按季度重新采样，获取均值  
# 或者求sum  
s7 = s7.resample("QS").sum()  
# 滑动窗口  
s7 = s7[s7[s7>0].rolling(window=3).mean()==3] # 大于零的数据，3个时间点，当前窗口的3个时间点都为true时，值为3。  
print(s7) # 输出结果  
  
s8 = pd.Series(np.random.randn(5), index=pd.date_range("2020-01-01", periods=5))  
# 从 2020-01-01 开始，5个时间点  
# 收益率  
print(s8.pct_change()) # 默认是1  
mask = s8.index.day==1 & s8.index.year==2020  
s9 = s8[mask] # 获取2020年的第一天的数据  
s10 = s8[~mask] # 获取非2020年的第一天的数据  
# 获取最高的5个数据  
print(s8.nlargest(5))
```

##### DataFrame
```python
import numpy as np  
import pandas as pd  
  
# 通过Series创建DataFrame  
s1 = pd.Series([1, 2, 3, 4, 5])  
s2 = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])  
s3 = pd.Series([1, 2, 3, 4, 5], name="series_name")  
df = pd.DataFrame({"col1": s1, "col2": s2, "col3": s3})  
print(df)  
  
# 通过字典创建  
df = pd.DataFrame(  
    {  
        "col1": [1, 2, 3, 4, 5],  
        "col2": [1, 2, 3, 4, 5],  
        "col3": [1, 2, 3, 4, 5]  
    }, index=["a", "b", "c", "d", "e"], columns=["col1", "col2", "col3"]  
)  
print(df)  
  
# 属性  
print("——————属性——————")  
print("索引", df.index)  
print("列表头", df.columns)  
print("二维列表值：", df.values)  
print("类型：", df.dtypes)  
print("形状：", df.shape)  
print("大小：", df.size)  
print("维度：", df.ndim)  
print("头：", df.head())  
print("尾：", df.tail())  
print("描述：", df.describe())  
print("信息", df.info())  
print("布尔是否为null（是false）", df.notnull())  
print("布尔是否为null（是true）", df.isnull())  
print("同上：", df.isna())  
# print("行列转置：", df.T)  
# 数学操作  
print("——————数学操作——————")  
# 统一计算每一列的运算，如果需要换轴（横着算），需要axis=1  
print("和：", df.sum())  
print("平均值：", df.mean())  
print("标准差：", df.std())  
print("方差：", df.var())  
print("最大值：", df.max())  
print("最小值：", df.min())  
print("中位数：", df.median())  
print("众数：", df.mode())  
print("0.25分位数：", df.quantile(0.25))  
# 获取键/值  
print("——————获取键/值——————")  
print("获取键：", df.keys())  
print("获取值：", df.values)  
print("获取索引：", df.index)  
# iloc、loc、iat、at  
print("——————索引——————")  
# 行数据获取，某一行  
print("第一行所有数据", df.iloc[0])  
print("第一行所有数据", df.loc["a"])  
# 列数据，某一列  
print("第一列的所有数据", df.iloc[:, 0]) # 第一列的所有数据  
print("第一列的所有数据", df.loc[:, "col1"]) # 第一列的所有数据  
# 单个数据  
print(df.iat[0, 0]) # 隐式二维索引  
print(df.at["a", "col1"]) # 显式索引  
# 其实loc也可以实现单个数据的索引  
print(df.loc["a", "col1"])  
print(df.iloc[0, 0])  
# 其他索引（一般是单列数据）  
print("__其他索引__")  
print(df["col1"]) # 这是个Series类型  
print(df.col1) # 也可以，直接 .列名，也是Series类型  
print(df[["col1"]]) # 也可以，就是麻烦，这个类型是DataFrame类型  
# 布尔筛选  
print("__布尔筛选__")  
print(df[df["col1"] > 2]) # 筛选大于2的每一行  
print(df[df > 2]) # 对整个DataFrame进行元素级筛选，保留所有大于2的元素，其余替换为NaN  
print(df[(df["col1"] > 2) & (df["col2"] > 3)]) # 取大于三的每一行  
print(df[(df["col1"] > 2) | (df["col2"] > 3)]) # 取大于二的每一行  
print(df[df.col1 > 2]) # 输出布尔值，大于2的为true  
# 随机取样  
print("__随机取样__")  
print(df.sample(n=2)) # 随机取样n行  
# 排序  
print("__排序__")  
print(df.sort_index())  
print(df.sort_values(by="col1", ascending=False)) # 比如有总分这一列，你就可以写总分  
print(df.nlargest(3, "col1")) # 取最大的3行
```

##### 文件操作
```python
import pandas as pd  
import json  
  
df_csv = pd.read_csv("complex_sales_data.csv")  
print(df_csv.head())  
print(df_csv.tail())  
  
df_csv.head().to_csv("new.csv") # 创建一个新文件，保存前五个  
  
print("--------JSON读取--------")  
"""  
df_json = pd.read_json("complex_sales_data_json.json")  
print(df_json.head())  
"""  
  
# 也可以这么读取JSON  
with open("complex_sales_data_json.json", "r", encoding="utf-8") as f:  
    data = json.load(f)  
df_json = pd.DataFrame(data)  
print(df_json.head())  
  
# 保存为CSV文件  
df_json.to_json("new_json.json", orient="records", indent=4, force_ascii=False)  
# 保存为JSON文件,  orient="records" 表示将DataFrame转换为JSON列表, indent=4 表示缩进4个空格,  
# force_ascii=False 表示将所有非ASCII字符转换为Unicode转义序列
```

##### 缺失值
```python
import numpy as np  
import pandas as pd  
  
print("--------缺失值--------")  
"""  
NaN是缺失值，“不是一个数字”，来自IEEE 754浮点数标准。在Pandas/NumPy中，它被实现为浮点数（float）类型。  
    这是它最著名的特性，NaN == NaN 的结果是False。  
    在数值型数据（int, float）中表示缺失值。  
NaT是时间戳的缺失值  
    Pandas为日期时间数据专门设计的缺失值标记。  
    类似于NaN，但是专门用于时间序列。  
NA是pandas的空值  
    Pandas（从1.0版开始）引入的通用缺失值标记，意图成为一个能覆盖所有数据类型的“一站式”缺失值。  
    它的行为比NaN更“友好”和一致。  
    但目前还处于推广阶段，NaN和NaT依然非常常见。  
None是错误或者不存在，表示“空”或“无”  
  
简单来说：  
Python的世界：None  
NumPy/Pandas的世界：NaN（Not a Number）， NaT（Not a Time）， NA（Not Available）  
NA = NaN(np) + NaT(pd)  
"""  
# s = pd.Series([1, np.nan, 3, None, pd.NA, pd.NaT])  
# df = pd.DataFrame({  
#     "col1": [1, np.nan, 3, None, pd.NA, pd.NaT],  
#     "col2": [1, np.nan, 3, None, pd.NA, pd.NaT],  
#     "col3": [1, np.nan, 3, None, pd.NA, pd.NaT]  
# })  
# print("是否是缺失值\n",s.isnull())  
# print("是否是缺失值\n",s.isna())  
# DataFrame一样的  
  
# 去掉缺失值  
# s = s.dropna()  
  
# df = df.dropna() # 横向去除一整条  
# 可以设置去除条件  
# df = df.dropna(how="all") # 当一整条均为空，才删除这一条记录  
# df = df.dropna(thresh=2) # 有n个不是缺失值就保留  
# 按列删除  
# df = df.dropna(axis=1)  
# 指定列进行检测  
# df = df.dropna(subset=["col1"]) # 检测col1列  
  
df = pd.read_csv("complex_sales_data.csv")  
print(df.isnull().sum(axis=0))  
# 填充  
df = df.fillna(0) # 填充0  
df = df.fillna({"客户评分":0}) # 填充指定列  
df = df.fillna(df[["客户评分"]].mean()) # 填充指定列的平均值  
df = df.ffill() # front fill，前一个值填充  
df = df.bfill() # back fill，后一个值填充  
print(df[["客户评分"]])
```

##### 重复值与类型转换
```python
import pandas as pd  
  
df = pd.DataFrame(  
    {  
        '姓名': ['张三', '李四', '王五', '张三', '赵六', '李四', '孙七', '王五'],  
        '年龄': [25, 30, 28, 25, 35, 30, 27, 28],  
        '城市': ['北京', '上海', '广州', '北京', '深圳', '上海', '杭州', '广州'],  
        '分数': [85, 92, 78, 85, 95, 92, 88, 78],  
        '部门': ['技术部', '销售部', '市场部', '技术部', '人事部', '销售部', '研发部', '市场部']  
    }  
)  
print(df)  
print(df.duplicated()) # 判断是否有重复行  
print(df.drop_duplicates()) # 删除重复行  
# 根据名称去重  
print(df.drop_duplicates(subset=['姓名']))  
# 保存最新的数据去重  
print(df.drop_duplicates(keep="last"))  
  
# 数据类型的转换  
print(df.dtypes)  
df[["分数"]] = df[["分数"]].astype("int32")  
print("转换后的：\n", df.dtypes)  
df[["部门"]] = df[["部门"]].astype("category") # 转换为分类型  
print("转换后的：\n", df.dtypes)
```

##### 数据变形
```python
import pandas as pd  
  
df = pd.DataFrame(  
    {  
        'ID': [1, 2, 3],  
        'Name': ['Alice Elysia', 'Bob Elysia', 'Charlie Elysia'],  
        'Math': [85, 92, 78],  
        'Science': [88, 90, 85],  
        'English': [92, 85, 88]  
    }  
)  
# df3["Name"]：返回 Series，适用于对单列进行数值计算、字符串操作等  
#  
# df3[["Name"]]：返回 DataFrame，适用于需要保持表格结构、选择多列或进行复杂数据处理的情况  
  
# 转置  
print(df.T)  
# 变换列，宽转窄  
df2 = pd.melt(df, id_vars=["ID", "Name"], var_name="科目", value_name="分数").sort_values("Name")  
print(df2)  
# 宽转高  
df3 = df2.pivot(index=["ID", "Name"], columns="科目", values="分数")  
print(df3)  
print("___________________")  
# 分列  
df[["Frist Name","Last Name"]] = df["Name"].str.split(" ", expand=True)  
# 以空格分裂名称，expand=True表示返回DataFrame  
# df["Name"].str.split(" ", expand=True)，Series转str转DataFrame，赋值给原来的DataFrame  
print(df)
```

##### 数据分箱
```python
import pandas as pd  
import numpy as np  
  
np.random.seed(42)  
  
data = {  
    'ID': range(1, 101),  
    'Age': np.random.randint(18, 80, 100),  # 年龄：18-79岁  
    'Income': np.random.normal(50000, 20000, 100).astype(int),  # 收入：正态分布  
    'Score': np.random.uniform(0, 100, 100),  # 分数：0-100均匀分布  
    'Height': np.random.normal(170, 10, 100),  # 身高：正态分布  
    'Weight': np.random.normal(70, 15, 100),  # 体重：正态分布  
    'Purchase_Amount': np.random.exponential(100, 100),  # 购买金额：指数分布  
    'Hours_Studied': np.random.randint(0, 50, 100),  # 学习时长：0-49小时  
    'Customer_Rating': np.random.uniform(1, 5, 100)  # 客户评分：1-5分  
}  
df = pd.DataFrame(data)  
df1 = pd.cut(df['Age'], bins=5, right=False)  
print(df1) # 分 5 段，左开右闭  
"""  
为什么是左开右闭？与数学惯例一致；累积分布函数（CDF）的自然表达；别问，问就是兼容与习惯  
来自知乎deephub用户的回答：  
在统计学里面做直方图或者频率分布表的时候，都是习惯用"小于等于"来描述边界。  
比如身高分组："150cm以下"、"150-160cm"、"160-170cm"，  
这里的"150-160cm"其实指的是"大于150、小于等于160"。  
如果你学过R语言（cut函数）、SPSS、SAS这些统计软件也都是这个逻辑，  
而Pandas就这样自然的继承了，因为Pandas的用户毕竟是统计学的人多，作为数据分析工具自然继承了这套约定。  
链接：https://www.zhihu.com/question/1973150943508968426/answer/1973178429345112471  
"""  
print(df1.value_counts()) # 统计5个分箱的数据  
# 自定义分箱范围  
df2 = pd.cut(df['Income'], bins=[0, 50000, 100000, 150000, 200000, np.inf]  
             , labels=['低', '中', '高', '很高', '非常高'])  
# 从0开始，到50000结束，到100000结束，到150000结束，到200000结束，到无穷大结束  
# labels桶标签  
# right=True，改为右闭右开  
print(df2)  
  
# qcut，等频率划分  
df3 = pd.qcut(df['Score'], q=4) # 4个分箱，默认左开右闭  
print(df3.value_counts())  
  
print("__________索引__________")  
df.set_index('ID', inplace=True)  
print(df) # 设置为 inplace=true 会修改原数据  
df.reset_index(inplace=True)  
print(df) # 恢复索引, 设置为 inplace=true 会修改原数据  
df.rename(columns={'ID': '编号'}, inplace=True, index={1: 'A'})  
print(df) # 重命名列和索引, 设置为 inplace=true 会修改原数据  
# 也可以使用 df.columns = ['编号','','',...] ，需要写全  
# df.index = [1,2,3,4,5,6,7,8,9,10...] ，需要写全
```

##### 时间数据处理
```python
import pandas as pd  
import numpy as np  
  
# 简单的时间  
d = pd.Timestamp("2023-01-01 10:22")  
d1 = pd.Timestamp("2023-01-01 12:22:00") # 和d是同一天  
print(d)  
print(d.year, d.month, d.day, d.hour, d.minute, d.second) # 打印时间  
print(d.quarter) # 获取季度  
print(d.weekday()) # 获取星期几  
print(d.day_name()) # 获取星期几  
print(d.days_in_month) # 获取月份天数  
# 判断  
print("————判断————")  
print(d.is_month_start) # 判断是否是月份开始  
print(d.is_month_end) # 判断是否是月份结束  
print(d.is_quarter_start) # 判断是否是季度开始  
print(d.is_quarter_end) # 是否是季度结束  
print(d.is_year_start) # 是否是年份开始  
print(d.is_year_end) # 是否是年份结束  
print(d.is_leap_year) # 是否是闰年  
  
# 转换  
print("————转换————")  
print(d.to_pydatetime()) # 转换为python时间  
print(d.to_datetime64()) # 转换为numpy时间，可以直接作用于DataFrame  
# 类似于：df["datatime"].dt.day_time()，获取时间属性并转化为星期  
# 因为获取的是Series对象，我们需要使用 dt 访问器转化为每个时间，才能对其进行day_name()  
print(d.to_numpy()) # 转换为numpy时间  
print(d.to_julian_date()) # 获取儒略日  
print(d.to_period("D"))  
print(d1.to_period("D")) # 和d是同一天  
# D , M , Y , W , Q  
  
# 读取是文件时候直接指定时间解析  
df = pd.read_csv("complex_sales_data.csv", parse_dates=["交易日期"]).head(10) # 将date解析为时间类型  
print(df.dtypes) # 查看交易日期的类型  
# 设置为索引  
df.set_index("交易日期", inplace=True)  
# 排序  
df = df.sort_index()  
# 排序后才可以切片  
print(df.loc["2023-01-13":"2023-03-13"])  
  
# 时间间隔  
print("————时间间隔————")  
d2 = pd.Timestamp("2023-01-01 10:22")  
d3 = pd.Timestamp("2023-01-01 12:22:00")  
t = d3 - d2  
print(t)  
print(type(t)) # Timedelta类型  
  
# 重新采样  
print("————重新采样————")  
df.dropna(inplace=True, subset=["利润", "客户评分"])  
print(df)  
print(df[["客户评分", "利润"]].resample("QE").mean()) # 季度平均重采样
```

##### 分类聚合
```python
import pandas as pd  
  
# 读取文件  
df = pd.read_csv("complex_sales_data.csv", parse_dates=["交易日期"]).head(10)  
# 缺失值统计  
print(df.isnull().sum())  
print(df.dtypes)  
# 分组  
df_group = df.groupby("地区")  
print(df_group.groups) # 获取分组  
# 或者查看某个具体的分组（按照城市分组）  
df_group_data = df_group.get_group("Central") # 拿到某个城市的数据  
print(type(df_group_data)) # 这是一个DataFrame  
print("这是利润\n", df_group_data["利润"]) # 获取利润  
print("这是利润平均值", df_group_data["利润"].mean()) # 获取平均值  
print("这是利润平均值", df_group_data["利润"].mean().round(2)) # 取两位小数  
  
# 多条件分组  
print("_____多条件分组_____")  
# 拿到(Central, Completed)两个共同条件的数据，这是一个元组  
df_M_group = df.groupby(["地区", "交易状态"]).get_group(("Central", "Completed"))  
print(type(df_M_group)) # 这是一个DataFrame  
print(df_M_group[["利润"]]) # 注意[["利润"]]还是一个DataFrame  
  
# 其他操作同理
```

### 数据库
#### MySQL（没人觉得这个 DB 很诡异么？）
```python
import pymysql  
from pymysql.connections import Connection  
  
def main() -> Connection:
    """  
        创建连接对象  
    """    # Connect to the database  
    con = pymysql.connect(  
        host='localhost',  
        user='root',  
        password='',  
        charset='utf8',  
        db='test'  
    )  
  
    # 创建游标对象  
    return con  
  
if __name__ == "__main__":  
    con = main() # 获取连接对象  
    cu = con.cursor()  
  
    try:  
        cu.execute("CREATE DATABASE IF NOT EXISTS aaa")  
        con.commit() # 提交创建，新增、修改、删除必须提交  
        print(cu.fetchall()) # 获取结果  
    except:  
        con.rollback() # 事务回滚  
  
    cu.execute("SHOW TABLES")  
    print(cu.fetchone())  # 一行一行获取结果  
    print(cu.fetchmany(2))  # 获取 2 行结果  
    print(cu.fetchall()) # 获取结果，这里游标已经没有结果了，所以只能获取一次  
  
    cu.execute(  
        "SELECT * FROM user WHERE name=%s and password=%s",('admin', 'admin')  
        )# 占位符，防止sql注入  
    print(cu.fetchone())  
    print(cu.fetchall())  
  
    # cu.execute(
    #         "SELECT * FROM user WHERE name=%(name)s and password=%(password)s",{  
    #             "name" : name,  
    #             "password" : password  
    #         }  
    #     )# 指名占位符，防止sql注入
  
    cu.close() # 关闭游标  
    con.close() # 关闭连接
```

#### MongoDB（MySQL 严父）
```python
import pymongo  
  
myclient = pymongo.MongoClient("mongodb://localhost:27017/")  
  
mydb = myclient["test"] # 创建数据库  
# 数据库只有在内容插入后才会创建! 就是说，数据库创建后要创建集合(数据表)并插入一个文档(记录)，数据库才会真正创建。  
  
dblist = myclient.list_database_names() # 获取所有数据库名称  
if "test" in dblist:  
  print("数据库已存在！")  
  
mydbMap = mydb["testMap"] # 创建集合  
# 集合只有在内容插入后才会创建! 就是说，创建集合(数据表)后要再插入一个文档(记录)，集合才会真正创建。  
  
collist = mydb.list_collection_names() # 获取所有集合名称  
if "sites" in collist:   # 判断 sites 集合是否存在  
  print("集合已存在！")  
  
print("-----------------插入--------------------")  
  
# 插入文档  
myData = {  
  "name": "John",  
  "address": "Highway 37"  
}  
x = mydbMap.insert_one(myData)  
print("文档插入成功，文档的 id 为：", x.inserted_id)  
  
# 插入列表map  
mylist = [  
    {"name": "Taobao", "alexa": "100", "url": "https://www.taobao.com"},  
    {"name": "QQ", "alexa": "101", "url": "https://www.qq.com"},  
    {"name": "Facebook", "alexa": "10", "url": "https://www.facebook.com"},  
    {"name": "知乎", "alexa": "103", "url": "https://www.zhihu.com"},  
    {"name": "Github", "alexa": "109", "url": "https://www.github.com"}  
]  
x = mydbMap.insert_many(mylist)  
print("列表插入成功，列表的 id 为：", x.inserted_ids)  
  
# 自定义 idmylist = [  
    {"_id": 1, "name": "Taobao", "alexa": "100", "url": "https://www.taobao.com"},  
    {"_id": 2, "name": "QQ", "alexa": "101", "url": "https://www.qq.com"},  
    {"_id": 3, "name": "Facebook", "alexa": "10", "url": "https://www.facebook.com"},  
]  
x = mydbMap.insert_many(mylist)  
print("列表插入成功，列表的 id 为：", x.inserted_ids)  
  
print("-----------------查询--------------------")  
  
# 查询第一个  
result = mydbMap.find_one()  
print(result)  
  
# 查询所有  
for x in mydbMap.find():  
    print(x)  
  
# 指定字段查询  
# filter：查询条件，是一个文档。如果为空 {}，则匹配集合中的所有文档。  
#  
# projection：投影，指定返回文档中应包含或排除哪些字段。1 表示包含，0 表示排除。  
# 投影文档不能混合使用包含和排除（_id 字段除外）  
#  
# options：其他选项，如排序、限制数量等。  
for x in mydbMap.find({}, {"_id": 0, "name": 1, "address": 1}):  
    print(x)  
  
# 批次查询  
for x in mydbMap.find().limit(2):  
    print(x)  
  
print("-----------------更新--------------------")  
# filter: 查询条件，用于匹配要更新的文档  
#  
# update: 更新操作，指定如何修改文档  
#  
# options (可选): 额外选项，如 upsertmyquery = {"address": "Highway 37"}  
values = {"$set": {"address": "Highway 37, New York"}}  
mydbMap.update_one(myquery, values) # 更新第一个匹配的文档  
print("文档更新成功")  
  
print("-----------------删除--------------------")  
# 删除单个  
myquery = { "name": "Taobao" }  
mydbMap.delete_one(myquery)  
  
# 批量删除  
# 删除以"Q"开头的  
result1 = mydbMap.delete_many({"name": {"$regex": "^Q"}})  
  
# 删除alexa 大于 80 的  
result2 = mydbMap.delete_many({"alexa": {"$gt": 80}})  
  
# $eq 等于 {"age": {"$eq": 25}}  
# $ne 不等于    {"age": {"$ne": 25}}  
# $gt 大于 {"age": {"$gt": 25}}  
# $gte    大于等于   {"age": {"$gte": 25}}  
# $lt 小于 {"age": {"$lt": 25}}  
# $lte    小于等于   {"age": {"$lte": 25}}  
# $in 在数组中   {"status": {"$in": ["active", "pending"]}}  
# $nin    不在数组中  {"status": {"$nin": ["inactive", "deleted"]}}  
  
# $and    逻辑与    {"$and": [{"age": {"$gt": 25}}, {"status": "active"}]}  
# $or 逻辑或    {"$or": [{"age": {"$lt": 18}}, {"age": {"$gt": 65}}]}  
# $not    逻辑非    {"age": {"$not": {"$lt": 18}}}  
# $nor    逻辑或非   {"$nor": [{"price": 1.99}, {"sale": true}]}  
  
# $exists 字段是否存在 {"email": {"$exists": true}}  
# $type   字段类型匹配 {"age": {"$type": "int"}}  
  
# $all    包含所有指定元素   {"tags": {"$all": ["mongodb", "python"]}}  
# $elemMatch  数组元素匹配条件   {"results": {"$elemMatch": {"$gte": 80, "$lt": 90}}}  
# $size   数组大小   {"tags": {"$size": 3}}  
  
# $regex  正则表达式匹配    {"name": {"$regex": "^张"}}  
# $text   文本搜索   {"$text": {"$search": "mongodb tutorial"}}  
# $expr   聚合表达式  {"$expr": {"$gt": ["$price", "$discount"]}}  
# $mod    取模运算   {"age": {"$mod": [2, 0]}}  
# $jsonSchema JSON模式匹配   {"$jsonSchema": {"bsonType": "object"}}
```

#### Faiss（超绝内存型向量数据库）

##### 简单线性索引
```python
import faiss  
import numpy as np  
  
# 简单索引  
np.random.seed(0) # 设置随机数种子  
  
def simple_index():  
    """  
    简单的线性索引  
    """    data = np.random.rand(10000, 256) # 随机生成10000行256列的数据  
    # 参数为索引维度  
    index = faiss.IndexFlatL2(256) # L2范数，欧式索引  
    """  
    index = faiss.IndexFlatIP(index) # 内积索引  
    index = faiss.index_factory(256, "Flat", faiss.METRIC_INNER_PRODUCT) # 创建内积索引，等效于11行  
    index = faiss.index_factory(256, "Flat", faiss.METRIC_L2) # 创建L2范数索引，等效于10行  
    """  
    index.add(data) # 添加数据  
  
    print(index.ntotal) # 索引的样本数量  
    print(index.d) # 索引的维度  
  
    # 创建新数据  
    query = np.random.rand(1, 256)  
    # 搜索近似的5个数据  
    index.search(query, 5)  
    # 解构数据  
    I, D = index.search(query, 5)  
    print(I)  
    print(D)  
  
    # 存储索引  
    faiss.write_index(index, "index.faiss")  
  
    # 删除数据  
    index.remove_ids(np.array([0,1,2,3]))  
    # 查看数据个数  
    print(index.ntotal)  
  
    # 删除所有向量数据  
    index.reset()  
    print(index.ntotal)  
  
if __name__ == "__main__":  
    simple_index()
```

##### 自定义索引 ID
```python
import faiss  
import numpy as np  
  
np.random.seed(0)  
  
def test01():  
    """  
    键映射  
    """    index_data = np.random.rand(1000000, 256)  
    index = faiss.IndexFlatL2(256) # 创建索引对象  
    index = faiss.IndexIDMap(index) # 升级为键映射索引对象  
    index.add_with_ids(index_data, np.arange(1000000,2000000))  # 添加数据，索引ID从1000000开始  
    print(index.ntotal) # 索引的样本数量  
    pass  
  
if __name__ == "__main__":  
    test01()
```

##### 聚类倒排索引
```python
import faiss  
import numpy as np  
  
np.random.seed(0) # 设置随机数种子  
  
def test01():  
    """  
    聚类倒排索引  
    """
    index_data = np.random.rand(10000, 256) # 随机生成1000000行256列的数据  
    qu = faiss.IndexFlatL2(256) # 创建索引对象  
    index = faiss.IndexIVFFlat(qu, 256, 100) # 升级为 IVF PQ 索引对象  
    """  
    第一个参数：基本索引对象  
    第二个参数：维度
    第三个参数：聚类的质心  
    """
    # 计算训练质心
    index.train(index_data) # 训练IVF索引  
    index.add(index_data) # 添加数据到索引对象  
    faiss.write_index(index, "IndexIVFFlat.faiss") # 写出文件
    print(index.ntotal)  
  
    query = np.random.rand(1, 256) # 创建查询数据  
    I, D = index.search(query, 5) # 搜索近似的 5个数据  
    print(I)  
    print(D)
  
if __name__ == '__main__':  
    test01()
```

##### 聚类倒排量化索引
```python
import faiss  
import numpy as np  
  
np.random.seed(0)  
  
def test01():  
    data = np.random.rand(10000, 256)  
    index = faiss.IndexFlatL2(256)  
    index = faiss.IndexIVFPQ(index, 256, 100, 32, 10)  
    """  
    1. 基本索引  
    2. 维度  
    3. 质心数  
    4. 桶数  
    5. 单向量划分数  
        比如：  
    数据矩阵长度256，个数10000，那么桶数就是划分10000，而单向量划分数就是划分长度256  
    一个向量划分为 n 份，n 越大，精度越高。划分最后是使用 8 位的数代替 256 精度  
    """    # 训练质心  
    index.train(data) # 训练IVF索引  
    index.add(data) # 添加数据  
    index.add_with_ids(data, np.arange(10000)) # 添加数据，索引ID从0开始  
    print(index.ntotal)  
    # 搜索  
    query = np.random.rand(1, 256)  
    index.nprobe = 3  # 设置搜索的质心数  
    index.search(query, 5)  
    I, D = index.search(query, 5)  
    print(I)  
    print(D)  
  
    faiss.write_index(index, "IndexIVFPQ.faiss")  
  
if __name__ == "__main__":  
    test01()
```

##### GPU 加速（仅限Linux平台）
```python
import faiss  
import numpy as np  
  
np.random.seed(0)  
  
def test01():  
    data = np.random.rand(10000, 256)  
    # 创建资源  
    res = faiss.StandardGpuResources()  
    # 在 CPU 创建索引  
    index_cpu = faiss.IndexFlatL2(256)  
    print(index_cpu)  
    # 将索引转到 GPU    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)  
    """  
    参数1：GPU 使用资源  
    参数2：GPU 设备编号  
    参数3：转移的索引  
    """    print(index_gpu)  
    # 插入数据  
    index_gpu.add(data)  
    # 向量搜索  
    D, I = index_gpu.search(np.random.rand(2, 256), k=2)  
    print(D)  
    print(I)  
  
if __name__ == "__main__":  
    test01()
```
