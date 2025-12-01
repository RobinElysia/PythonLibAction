## 环境配置

准备工作：
1. 虚拟机：WSL Ubuntu （建议22）
2. Python：3.10
3. [virtualenv：任意版本](https://pypi.org/project/virtualenv/#files)
4. [CUDA：11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0)
5. [cuDNN：CUDA 11.X → cuDNN 8.9.7](https://developer.nvidia.com/rdp/cudnn-archive)
6. [PyTorch：2.0.1](https://pypi.org/project/torch/2.0.1/#files)
7. [Tensorflow：2.12.1](https://pypi.org/project/tensorflow/2.12.1/#files)

注意：Python、CUDA、cuDNN、PyTorch、Tensorflow这五者拥有较强的依赖关系，其中 CUDA and cuDNN 是强依赖关系，CUDA 与 Pytorch、Tensorflow 是强依赖关系。因为我这里使用的是离线安装，很大程度上需要自行处理依赖关系，所以特此强调。对于直接 pip install XXX 的在线用户，大概率无需考虑 pytorch 和 tensorflow 的版本问题。

### 虚拟环境与 Python 安装

这里采用 WSL2 作为本地开发环境支持，为什么不选 VM 和 其他的虚拟化技术，一是 VM 有点大材小用的感觉，虚拟化环境配发 4 核心或许不太够用；为什么不用 Docker 虚拟环境，一方面是出现的各种硬件虚拟化问题，一方面是精简虚拟化环境可能导致各种意料之外的问题。所以我们采用折中方案， WSL2 虚拟化。

对于 WSL2 而言，我建议使用 Ubuntu 20/22 版本，24 版本不支持cuda11.8和对应的cudnn。

#### 手动编译 Python 环境
因为apt不支持安装特定版本，所以这里需要去 python 官网或者 ustc 源下载高版本的压缩包，解压缩编译
这里我们选择 ustc 源 进行下载，链接如下：
https://mirrors.ustc.edu.cn/python/3.10.8/Python-3.10.8.tar.xz
https://mirrors.ustc.edu.cn/python/3.11.14/Python-3.11.14.tar.xz
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
sudo ./configure --prefix=/usr/local/python3.10 \
                 --enable-optimizations \
                 --with-ensurepip=install
                 

sudo make -j$(nproc)
sudo make install

# 有时候需要进行链接，如果查看python3.10不管用的话
sudo ln -sf /usr/local/python3.10/bin/python3.10 /usr/bin/python3.10

# 同理Python3.11如法炮制
# 编译完成就成功了

```

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
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8
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

# 运行提示命令
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.3.28/cudnn-*-keyring.gpg /usr/share/keyrings/

# 更新仓库
sudo apt-get update

# 安装
sudo apt-get install -y libcudnn8 libcudnn8-dev libcudnn8-samples

```

### 安装 Virtualenv、Pytorch、Tensorflow、Ollama、Model

#### Virtualenv

```bash

# 进入 opt
cd /opt
# 下载 whl
wget https://files.pythonhosted.org/packages/79/0c/c05523fa3181fdf0c9c52a6ba91a23fbf3246cc095f26f6516f9c60e6771/virtualenv-20.35.4-py3-none-any.whl # 需自行找下载链接，在上文我给到了

# 安装
pip install /opt/virtualenv-20.35.4-py3-none-any.whl

```

以下是直接在全局进行安装 Pytorch Tensorflow

#### Pytorch

```bash

# 在 opt 下
wget https://files.pythonhosted.org/packages/21/33/4925decd863ce88ed9190a4bd872b01c146243ee68db08c72923984fe335/torch-2.0.1-cp310-cp310-manylinux2014_aarch64.whl # 需自行找下载链接，在上文我给到了

# 安装
pip install /opt/torch-2.0.1-cp310-cp310-manylinux2014_aarch64.whl

```

#### Tensorflow

```bash

# 在 opt 下
wget https://files.pythonhosted.org/packages/21/33/4925decd863ce88ed9190a4bd872b01c146243ee68db08c72923984fe335/torch-2.0.1-cp310-cp310-manylinux2014_aarch64.whl # 需自行找下载链接，在上文我给到了

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

关于`Ollama Modelfile`和部分Modelfile中的系统调试，你可以参考后续的ModelFile文件编写

### OpenWebUI 安装（可选）

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

因为我使用的是权威的 `uv` 项目管理器，所以直接给出 `toml` ：

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
    "lxml==6.0.2",          # C 级高性能 XML/HTML 解析器，支持 XPath
    "BeautifulSoup4==4.12.3", # 对不规则 HTML 友好的高层解析、搜索、遍历接口

    # 数据分析 - 结构化数据二维表格化操作与统计
    "pandas==2.2.3",        # 提供 DataFrame/Series，支撑读写、过滤、聚合、透视等全套 EDA 流程
    "pyarrow==17.0.0",

    # 数据库 - 与 MySQL、MongoDB、向量检索引擎交互的驱动与 ORM
    "pymysql==1.1.1",       # 纯 Python 实现的 MySQL 客户端
    "pymongo==4.15.4",       # 官方 MongoDB 驱动
    "faiss-cpu==1.9.0",     # Facebook 开源的稠密向量相似度检索库（CPU 版）
    "SQLAlchemy==2.0.36",    # Python 事实标准 ORM & SQL 工具链
    "mysqlclient",   # SQLAlchemy链接mysql需要，和pymysql都可以

    # 机器学习基本工具 - 经典 ML+可视化+数值计算
    "numpy>=1.25.0", # 张量/向量化计算基石，sklearn、torch 均依赖
    "scikit-learn==1.7.2",   # 传统机器学习算法（分类/回归/聚类/降维/预处理）
    "matplotlib",     # 2D 静态可视化，科研作图首选

    # 深度学习库 - Transformer 生态、PyTorch 全家桶及配套评估/加速/日志组件
    "transformers==4.45.2",   # Hugging Face 社区预训练 SOTA 模型库（BERT/GPT/T5…）
    "datasets==2.18.0",       # HF 社区 1000+ 标准/自定义数据集加载、缓存与处理接口
    "evaluate",       # HF 统一封装 GLUE、ROUGE、BLEU 等常用评估指标
    "peft==0.10.0",           # 参数高效微调（LoRA/AdaLoRA/P-tuning）官方实现
    "accelerate==0.27.0",     # HF 分布式训练/混合精度/CPU offload 通用框架
    "optimum",        # 针对 Intel/ONNX/OpenVINO/NVIDIA 的推理加速与量化工具箱
    "sentencepiece==0.1.99",  # Google 子词分词器（支持 BPE/Unigram）
    "nltk==3.9.1",           # 经典 NLP 工具集（分句、词性、情感词典等）
    "torch==2.4.0+cu118",     # PyTorch GPU 2.2.0（CUDA 11.8）核心库
    "torchvision==0.19.0+cu118", # 官方视觉模型/数据变换/数据集
    "torchaudio==2.4.0+cu118", # 官方语音模型/特征提取/数据集
    "tqdm==4.66.5",           # 进度条美化，训练/数据加载可视化
    "tensorboard==2.13.0",    # 训练日志可视化（兼容 PytorchLib-lightning、transformers）
    "torchsummary==1.5.1",

    # 页面开发
    "watchdog",
    "streamlit"
]

[[tool.pip.index-url]]
url = "https://pypi.org/simple"

[[tool.uv.index]]
url = "https://download.pytorch.org/whl/cu118"

[tool.setuptools]
packages = []      # 只装依赖，不打包任何源码
```
