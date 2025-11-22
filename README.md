# PythonLibAction！  
💡 探索、学习、贡献！这是一个开放的 Python AI 第三方库学习社区。我们不仅收集和解析最酷的 Python AI 库，还鼓励你通过提交代码和案例来共同构建这份知识图谱。一起点亮 Python AI 技能树吧！

## 目录
- [环境配置](#环境配置)
  - [虚拟环境与 Python 安装](#虚拟环境与-python-安装)
    - [高版本系统](#高版本系统)
    - [低版本系统](#低版本系统)
  - [CUDA 与 cuDNN](#cuda-与-cudnn)
    - [CUDA](#cuda)
    - [cuDNN](#cudnn)
  - [安装 Virtualenv、Pytorch、Tensorflow、Ollama、Model](#安装-virtualenvpytorchtensorflowollamamodel)
    - [Virtualenv](#virtualenv)
    - [Pytorch](#pytorch)
    - [Tensorflow](#tensorflow)
    - [Virtualenv 虚拟环境下安装环境（不如 uv）](#virtualenv-虚拟环境下安装环境不如-uv)
    - [Ollama、模型安装](#ollama模型安装)
  - [OpenWebUI 安装](#openwebui-安装)
  - [其他](#其他)

- [慢速开始](#慢速开始)
  - [Linux 使用技巧](#linux-使用技巧)
    - [Vim](#vim)
  - [正则表达式（讨厌re的原因是因为它是设计给天才看的）](#正则表达式讨厌re的原因是因为它是设计给天才看的)
    - [实践](#实践)
  - [Ollama Modelfile 权威指南](#ollama-modelfile-权威指南)
    - [1. 简介：什么是 Modelfile](#1-简介什么是-modelfile)
    - [2. Modelfile 的基本结构](#2-modelfile-的基本结构)
    - [3. 常用字段详解](#3-常用字段详解)
      - [FROM](#from)
      - [TEMPLATE](#template)
      - [PARAMETER](#parameter)
      - [STOP](#stop)
    - [4. 模板（TEMPLATE）与停止符（STOP）](#4-模板template与停止符stop)
    - [5. 参数（PARAMETER）详解与推荐值](#5-参数parameter详解与推荐值)
    - [6. 常见使用场景示例](#6-常见使用场景示例)
      - [6.1 简单推理服务（聊天机器人）](#61-简单推理服务聊天机器人)
      - [6.2 指令微调（SFT）后的模型部署](#62-指令微调sft后的模型部署)
      - [6.3 大模型量化与低内存部署](#63-大模型量化与低内存部署)
    - [7. 附录：示例 Modelfile 热门变体](#7-附录示例-modelfile-热门变体)
      - [最小可用 Modelfile](#最小可用-modelfile)
      - [完整的 Demo](#完整的-demo)
  - [数据处理](#数据处理)
    - [JSON（万物基于JSON）](#json万物基于json)
      - [代码](#代码)
      - [测试 JSON 文件](#测试-json-文件)
    - [XML（谁？不熟，感觉是Java佬最爱）](#xml谁不熟感觉是java佬最爱)
      - [XML 数据清洗](#xml-数据清洗)
      - [测试 XML 文件](#测试-xml-文件)
    - [HTML（BeautifulSoup，漂亮的汤（](#htmlbeautifulsoup漂亮的汤)
      - [HTML 数据清洗](#html-数据清洗)
      - [测试 HTML 文件](#测试-html-文件)
    - [pandas（你为什么不学熊猫？）](#pandas你为什么不学熊猫)
      - [样本数据生成](#样本数据生成)
      - [Series](#series)
      - [DataFrame](#dataframe)
      - [文件操作](#文件操作)
      - [缺失值](#缺失值)
      - [重复值与类型转换](#重复值与类型转换)
      - [数据变形](#数据变形)
      - [数据分箱](#数据分箱)
      - [时间数据处理](#时间数据处理)
      - [分类聚合](#分类聚合)
  - [数据库](#数据库)
    - [MySQL（没人觉得这个 DB 很诡异么？）](#mysql没人觉得这个-db-很诡异么)
    - [MongoDB（MySQL 严父）](#mongodbmysql-严父)
    - [Faiss（超绝内存型向量数据库）](#faiss超绝内存型向量数据库)
      - [简单线性索引](#简单线性索引)
      - [自定义索引 ID](#自定义索引-id)
      - [聚类倒排索引](#聚类倒排索引)
      - [聚类倒排量化索引](#聚类倒排量化索引)
      - [GPU 加速（仅限Linux平台）](#gpu-加速仅限linux平台)
    - [SQLAlchemy](#sqlalchemy)
      - [简单的查询](#简单的查询)
      - [复杂的查询](#复杂的查询)
      - [ORM 的一切](#orm-的一切)
  - [机器学习/深度学习](#机器学习深度学习)
    - [Tensorflow](#tensorflow)
    - [Pytorch](#pytorch-1)
      - [前置知识](#前置知识)
      - [Pytorch Lib](#pytorch-lib)
    - [Transformers](#transformers)
      - [Pipline](#pipline)
      - [Tokenizer](#tokenizer)
      - [Easy Model](#easy-model)
      - [ComModel](#commodel)
      - [Datasets](#datasets)
    - [sk-learn](#sk-learn)
    - [peft](#peft)
    - [FastAPI](#fastapi)
    - [LangChain](#langchain)

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
export PATH=/usr/local/cuda-11.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}
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

# dpkg：Debin package
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

# 安装open-webui
pip install /opt/openwebui/open_webui-0.6.36-py3-none-any.whl

# 安装高版本sqlite3
# 所有的Ubuntu都需要，因为自带的系统没有sqlite3.35以上版本

# 下载
wget https://sqlite.org/2025/sqlite-autoconf-3510000.tar.gz

# 解压缩
tar -vxf sqlite-autoconf-3510000.tar.gz

# Debian/Ubuntu:
apt update && apt install -y build-essential gcc make tcl

# 配置编译选项（推荐启用 FTS5、JSON、RTree）
./configure \
  --prefix=/usr/local \
  --enable-fts5 \
  --enable-rtree \
  --enable-shared \
  --enable-static

# 编译
make -j$(nproc)

# 安装（可选，但建议，安装到系统）
make install

# 验证sqlite版本
/usr/local/bin/sqlite3 --version

# 修改对应的init文件（偷梁换柱）
# 在：site-packages/chromadb/__init__.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# 运行
open-webui serve

```

### 其他
标准库：json、os、xml和re无需下载，直接导入使用
第三方库：
1. pymysql：直接pip install就可以
2. pymongo：同上
3. faiss：Windows平台仅支持cpu版本，py版本在3.14 ~ 9（我使用的是3.10.X版本）；gpu加速版本仅限Linux平台，版本限制在3.10 ~ 6。官方建议在conda虚拟环境上安装，但是没必要，我将使用权威的 uv。
4. sqlite3：仅限Linux平台，自己搭建可能需要手动编译>=3.35.0版本，编译完成后重新让Py链接sqlite3的依赖（这部分如果你的py也是编译的，那就重新编译一遍）
5. BeautifulSoup：使用4.X版本
	1. 需要下载 lxml
6. SQLAlchemy：2.0.44版本的，尽量使用2.X版本

因为我使用的是权威的 `uv` 项目管理器，所以直接给出 `toml` 好吧：
```toml
# pyproject.toml  
[build-system]  
requires = ["setuptools>=61", "wheel"]  
build-backend = "setuptools.build_meta"  
  
[project]  
name = "mysqlproject"  
version = "0.1.0"  
description = "Add your description here"  
requires-python = "==3.10.11"  
dependencies = [  
    # 解析/爬虫 - 快速、容错地抽取/清洗 HTML/XML 文本  
    "lxml",          # C 级高性能 XML/HTML 解析器，支持 XPath    "BeautifulSoup4", # 对不规则 HTML 友好的高层解析、搜索、遍历接口  
  
    # 数据分析 - 结构化数据二维表格化操作与统计  
    "pandas",        # 提供 DataFrame/Series，支撑读写、过滤、聚合、透视等全套 EDA 流程  
  
    # 数据库 - 与 MySQL、MongoDB、向量检索引擎交互的驱动与 ORM    "pymysql",       # 纯 Python 实现的 MySQL 客户端  
    "pymongo",       # 官方 MongoDB 驱动  
    "faiss-cpu",     # Facebook 开源的稠密向量相似度检索库（CPU 版）  
    "SQLAlchemy",    # Python 事实标准 ORM & SQL 工具链  
    "mysqlclient",   # SQLAlchemy链接mysql需要，和pymysql都可以  
  
    # 机器学习基本工具 - 经典 ML+可视化+数值计算  
    "numpy>=1.23,<2", # 张量/向量化计算基石，sklearn、torch 均依赖  
    "scikit-learn",   # 传统机器学习算法（分类/回归/聚类/降维/预处理）  
    "matplotlib",     # 2D 静态可视化，科研作图首选  
  
    # 深度学习库 - Transformer 生态、PyTorch 全家桶及配套评估/加速/日志组件  
    "transformers",   # Hugging Face 社区预训练 SOTA 模型库（BERT/GPT/T5…）  
    "datasets",       # HF 社区 1000+ 标准/自定义数据集加载、缓存与处理接口  
    "evaluate",       # HF 统一封装 GLUE、ROUGE、BLEU 等常用评估指标  
    "peft",           # 参数高效微调（LoRA/AdaLoRA/P-tuning）官方实现  
    "accelerate",     # HF 分布式训练/混合精度/CPU offload 通用框架  
    "optimum",        # 针对 Intel/ONNX/OpenVINO/NVIDIA 的推理加速与量化工具箱  
    "sentencepiece",  # Google 子词分词器（支持 BPE/Unigram）  
    "nltk",           # 经典 NLP 工具集（分句、词性、情感词典等）  
    "rouge",          # 摘要/翻译评估 ROUGE-N/L/SU 指标独立包  
    "torch==2.2.0+cu118",     # PyTorch GPU 2.2.0（CUDA 11.8）核心库  
    "torchvision==0.17.0+cu118", # 官方视觉模型/数据变换/数据集  
    "torchaudio==2.2.0+cu118", # 官方语音模型/特征提取/数据集  
    "pillow>=9.0",   # 轻量级图像 I/O 与基础变换（torchvision 前置依赖）  
    "tqdm",           # 进度条美化，训练/数据加载可视化  
    "tensorboard",    # 训练日志可视化（兼容 pytorch-lightning、transformers）  
    "scipy"           # 稀疏矩阵、优化、统计分布等科学计算补充  
]  
  
[[tool.pip.index-url]]  
url = "https://pypi.org/simple"  
  
[[tool.uv.index]]  
url = "https://download.pytorch.org/whl/cu118"  
  
[tool.setuptools]  
packages = []      # 只装依赖，不打包任何源码
```

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

#### SQLAlchemy
##### 简单的查询
```python
import sqlalchemy  
from sqlalchemy.sql import and_, or_, not_  
  
en = sqlalchemy.create_engine("mysql://root:123456@localhost/testdb")  
  
# 创建元数据  
meta_data = sqlalchemy.MetaData()  
  
# 创建表  
person = sqlalchemy.Table(  
    "person", meta_data,  
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),  
    sqlalchemy.Column("name", sqlalchemy.String(32)),  
    sqlalchemy.Column("age", sqlalchemy.Integer)  
)  
"""  
name: 表名  
meta_data: 元数据  
Column: 字段信息  
"""  
# 创建表  
meta_data.create_all(en)  
  
print("_____插入_____")  
  
# 插入一个记录  
insert = person.insert().values(name="张三", age=18)  
with en.connect() as con: # 创建连接  
    result = con.execute(insert) # 执行插入  
    print(result.inserted_primary_key) # 获取插入的id  
    con.commit() # 提交  
  
# 多条记录  
insert = person.insert()  
with en.connect() as con:  
    result = con.execute(insert, [  
        {"name": "张三", "age": 18},  
        {"name": "张三", "age": 12},  
        {"name": "张三", "age": 20}  
    ])  
    print(result.inserted_primary_key)  
    con.commit()  
  
print("_____查询_____")  
# 普通查询  
select = person.select()  
with en.connect() as con:  
    result = con.execute(select)  
    for row in result:  
        print(row) # 这是一个元组  
    # 或者直接提取数据，结果集大不建议  
    fetchResult = result.fetchall()  
    print(fetchResult)  
    # 取一个数据  
    fetchOne = result.fetchone()  
    print(fetchOne)  
  
# 条件查询  
select = person.select().where("张三" == person.c.name).where(person.c.age > 18) # 创建查询条件  
with en.connect() as con:  
    result = con.execute(select)  
    for row in result:  
        print(row)  
  
# 带符号的条件的查询  
select = person.select().where(and_(person.c.name == "张三", person.c.age > 18)) # 两个条件and  
with en.connect() as con:  
    result = con.execute(select)  
    for row in result:  
        print(row)  
  
# 更新  
# select = person.update().values(name="李四") # 全部全部为李四  
select = person.update().where(1 == person.c.id).values(name="王五")  
with en.connect() as con:  
    result = con.execute(select)  
    print(result.rowcount) # 影响的行数  
    con.commit()  
  
# 删除  
select = person.delete().where(1 == person.c.id) # 删除id为1的  
with en.connect() as con:  
    result = con.execute(select)  
    print(result.rowcount)  
    con.commit()
```

##### 复杂的查询
```python
from typing import Tuple  
  
import sqlalchemy  
from sqlalchemy.sql import and_  
  
def create_en() -> Tuple[sqlalchemy.engine.base.Engine, sqlalchemy.MetaData]:  
    en = sqlalchemy.create_engine("mysql://root:123456@localhost/testdb")  
  
    # 创建元数据  
    meta_data = sqlalchemy.MetaData()  
    return en, meta_data  
  
def create_tb(meta_data) -> Tuple[sqlalchemy.Table, sqlalchemy.Table]:  
    # 部门  
    department = sqlalchemy.Table(  
        "department", meta_data,  
        sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),  
        sqlalchemy.Column("name", sqlalchemy.String(32))  
    )  
  
    # 员工  
    employee = sqlalchemy.Table(  
        "employee", meta_data,  
        sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),  
        sqlalchemy.Column("name", sqlalchemy.String(32)),  
        sqlalchemy.Column("age", sqlalchemy.Integer),  
        # 外键  
        sqlalchemy.Column("department_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("department.id"))  
    )  
    return department, employee  
  
def insert_data(en, department, employee):  
    with en.connect() as con:  
        con.execute(department.insert(), [  
            {"name": "it"},  
            {"name": "hr"}  
        ])  
        con.execute(employee.insert(), [  
            {"name": "张三", "age": 18, "department_id": 1},  
            {"name": "李四", "age": 12, "department_id": 1},  
            {"name": "王五", "age": 20, "department_id": 2}  
        ])  
        con.commit()  
  
def select_data(en, department, employee):  
    with en.connect() as con:  
        # 联合查询  
        join = employee.join(department, department.c.id == employee.c.department_id)  
        # q = sqlalchemy.select(join).where(and_(department.c.name == "it")) # 创建查询条件  
        q = sqlalchemy.select(employee).select_from(join).where(and_(department.c.name == "it")) # 不带部门  
        # 执行查询  
        result = con.execute(q)  
        for row in result:  
            print(row)  
  
if __name__ == "__main__":  
    # 创建连接和元数据  
    en, meta_data = create_en()  
  
    # 创建表  
    department, employee = create_tb(meta_data)  
  
    # 查询  
    select_data(en, department, employee)  
  
    # 插入数据  
    meta_data.create_all(en)
```

##### ORM 的一切
```python
from typing import Type, List  
from sqlalchemy import Engine, insert, select  
from sqlalchemy.sql import and_  
from sqlalchemy.ext.declarative import declarative_base  
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session  
  
import sqlalchemy  
  
def create_en() -> tuple[Engine, Type[DeclarativeBase], sessionmaker[Session]]:  
    en = sqlalchemy.create_engine("mysql://root:123456@localhost/testdb")  
  
    # 创建基类  
    Base = declarative_base()  
  
    # 创建会话  
    Session = sessionmaker(bind=en)  
    return en, Base, Session  
  
# 创建连接  
en, Base, Session = create_en()  
  
class Person(Base):  
    __tablename__ = "person"  
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)  
    name = sqlalchemy.Column(sqlalchemy.String(32))  
    age = sqlalchemy.Column(sqlalchemy.Integer)  
  
def insert_data(person: Person, session: sessionmaker[Session], Base: Type[DeclarativeBase]):  
    # 创建会话  
    insert_session = session()  
    # 添加数据  
    insert_session.add(person)  
    # 提交数据  
    insert_session.commit()  
  
def insert_M_data(person: List[Person], session: sessionmaker[Session], Base: Type[DeclarativeBase]):  
    # 创建会话  
    insert_session = session()  
    # 批量添加数据  
    insert_session.add_all(person)  
    # 提交数据  
    insert_session.commit()  
  
def select_data(person: Person, en: Engine, session: sessionmaker[Session], Base: Type[DeclarativeBase]):  
    # 创建会话  
    select_session = session()  
    # 查询所有  
    result = select_session.query(person).all()  
    for row in result:  
        print(row)  
  
def select_con_data(person: Person, session: sessionmaker[Session], Base: Type[DeclarativeBase]):  
    # 创建会话  
    select_session = session()  
    # 条件查询第一个，查询可以为空  
    result = select_session.query(person).filter(and_(person.name == "张三")).first()  
    # 结果集只有一条记录使用one，查询不能为空  
    # result = select_session.query(person).filter(and_(person.name == "张三")).one()  
    # 结果集只有一条记录，但是可以为空  
    # result = select_session.query(person).filter(and_(person.name == "张三")).scalar()  
    print(result)  
  
def update_base_query_data(person: Person, session: sessionmaker[Session], Base: Type[DeclarativeBase]):  
    # 创建会话  
    update_session = session()  
    # 更新  
    result = update_session.query(person).one()  
    # 修改  
    result.name = "赵四"  
    # 提交数据  
    update_session.commit()  
  
def update_func_data(person: Person, session: sessionmaker[Session], Base: Type[DeclarativeBase]):  
    # 创建会话  
    update_session = session()  
    # 更新  
    result = update_session.query(person).update({"name": "小潘"})  
    # 提交数据  
    update_session.commit()  
  
# 批量插入  
def insert_M_useValue_data(session: sessionmaker[Session]):  
    # 创建会话  
    insert_session = session()  
    # 批量添加数据  
    insert_session.execute(  
        insert(Person).values([  
            {"id": 4, "name": "王五", "age": 18},  
            {"id": 5, "name": "赵刘", "age": 18},  
        ])  
    )  
    # 提交数据  
    insert_session.commit()  
  
# 嵌套查询的批量插入  
def insert_M_useSelect_data(session: sessionmaker[Session]):  
    # 创建会话  
    insert_session = session()  
    insert_session.execute(  
        insert(Person).values(  
            [  
                {  
                    "id": 6, "name": "bob", "age": select(Person.age).where(Person.id == 1)  
                },  
                {  
                    "id": 7, "name": "lili", "age": select(Person.age).where(Person.id == 2)  
                },  
            ]  
        )  
    )# 查询id为1 2的年龄作为6 7年龄  
  
# 更新删除同理也可以使用execute进行批量  
  
# 什么？你问我事务和多数据源？事务天然支持，多数据源一行代码搞定  
# with Session(engine) as session1, session1.begin(), Session(engine2) as session2.begin():  
  
if __name__ == "__main__":  
    # 插入数据  
    person = Person(id=1, name="张三", age=18)  
    insert_data(person, Session, Base)  
  
    # 批量插入  
    ps = [  
        Person(id=2, name="王五", age=18),  
        Person(id=3, name="李四", age=18),  
    ]  
    insert_M_data(ps, Session, Base)  
  
    # 查询  
    person = Person(id=1)  
    select_data(person, en, Session, Base)  
  
    # 条件查询  
    person = Person(id=1)  
    select_con_data(person, Session, Base)  
  
    # 修改1  
    person = Person(id=1)  
    update_base_query_data(person, Session, Base)  
  
    # 修改2  
    person = Person(id=1)  
    update_func_data(person, Session, Base)  
  
    # 批量插入  
    insert_M_useValue_data(Session)  
  
    # 嵌套查询的批量插入  
    insert_M_useSelect_data(Session)  
  
    # 插入数据  
    Base.metadata.create_all(en)
```

### 机器学习/深度学习
#### Tensorflow

#### Pytorch
##### 前置知识
参考[Perceptron_PytorchLib 项目目录](https://github.com/RobinElysia/PythonLibAction/tree/main/pytorch/Perceptron_PytorchLib)，包含手操微分、手写各种损失函数、激活函数、各种梯度算法、三层网络和BP反向传播等等。

##### Pytorch Lib

#### Transformers
##### Pipline
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

##### Tokenizer
```python
from transformers.pipelines import SUPPORTED_TASKS  
from transformers import pipelines, AutoModelForSequenceClassification, AutoTokenizer  
import torch  
  
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
        offset_mapping = [(0,1), (2,6), (7,9)]        如果模型告诉你“第 2 个 token 是答案”，你就知道答案是原文 2:6 → "love"。  
    """    print(input.word_ids) # 获取单词的索引  
    """  
    例子：  
        原文："ChatGPT is amazing"  
        分词后：["Chat", "##G", "##PT", "is", "amazing"]  
        word_ids() = [0, 0, 0, 1, 2]        你就知道前 3 个子词都属于第 0 号单词，后面依次是第 1、2 号单词。  
    """    # 这两个是Fast独有的  
  
    # SlowTokenizer  
    tokenizer_slow = AutoTokenizer.from_pretrained(  
        "uer/roberta-base-finetuned-dianping-chinese", use_fast=False  
    )  
    # 有的模型可能不支持FastTokenizer，如果你不支持，那就受着吧  
  
    # 特殊的  
    # 有的模型需要在远程进行加载，比如远程加载模型，需要在创建对象的时候加上属性：trust_remote_code=True  
  
if __name__ == "__main__":  
    Easy_Tokenizer()
```

##### Easy Model
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
            通常返回的是最后一层的隐藏状态向量        带Head的模型（如BertForSequenceClassification、AutoModelForSequenceClassification）：  
            包含骨干网络 + 任务特定的头部（Head）  
            在骨干网络基础上添加了针对特定任务的输出层            直接输出任务相关的结果（如分类logits）  
        它们前置任务类似：            输入处理：两者都使用相同的tokenizer进行分词  
            嵌入层：都经过相同的词嵌入、位置嵌入等  
            Transformer编码器：都通过相同的多层Transformer结构  
            特征提取：都得到相同质量的上下文表示        关键差异在输出阶段，Head通常是一个或多个线性层（Linear Layer）  
    """  
if __name__ == "__main__":  
    Easy_Model()
```

##### ComModel
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

##### Datasets
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
    # datasets = load_dataset("super_glue", "boolq", split = "train[0:100]")    return datasets  
  
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
        """        model_inputs = tokenizer(examples["content"], max_length=512, truncation=True)  
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
        """        model_inputs = tokenizer(examples["content"], max_length=512, truncation=True) # 创建新的变量保存分词  
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
    data = Easy_Datasets_DataLoad()  
    Easy_Opration_Datasets(data)
```

#### sk-learn

#### peft

#### FastAPI

#### LangChain
