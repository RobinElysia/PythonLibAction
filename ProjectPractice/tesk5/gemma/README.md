# TourTools LP - 智能旅游推荐系统

基于 Gemma2 大语言模型和 RAG 技术的智能旅游推荐系统，根据用户的地貌偏好推荐国内旅游目的地，并提供智能问答服务。

## 功能特性

- **地貌偏好分析**：用户可对山川、海洋、平原三种地貌类型进行 0-10 分评分
- **智能推荐**：基于偏好分数推荐最匹配的旅游目的地（山西/海南/北京）
- **RAG 增强问答**：使用 FAISS 向量检索和知识库，回答目的地相关问题
- **图文展示**：从数据库获取并展示目的地图片
- **对话历史**：保存完整的对话记录，支持连续交互

## 技术栈

- **前端框架**：Streamlit
- **AI 模型**：
  - Gemma2（主对话模型）
  - BGE-base-zh-v1.5（中文向量嵌入模型）
- **向量检索**：FAISS
- **数据库**：MySQL (PyMySQL)
- **其他**：PyTorch, Transformers, BeautifulSoup4

## 项目结构

```
.
├── main.py                 # 应用入口
├── config.py              # 配置文件
├── requirements.txt       # 依赖列表
├── prompt.txt            # LLM 提示词模板
├── re_zone.py            # 清理工具脚本
├── models/               # AI 模型加载
│   ├── llm_model.py     # Gemma2 模型
│   └── rag_model.py     # RAG 和向量检索
├── services/            # 业务逻辑
│   ├── chat.py         # 对话生成
│   └── recommendation.py # 推荐逻辑
├── ui/                  # UI 组件
│   ├── sidebar.py      # 侧边栏
│   └── chat.py         # 聊天界面
├── utils/              # 工具函数
│   ├── db_utils.py    # 数据库操作
│   └── image_utils.py # 图片处理
└── data/               # 数据文件
    ├── data.html      # 景点知识库
    └── tour.faiss     # FAISS 索引（自动生成）
```

## 安装部署

### 环境要求

- Python 3.8+
- CUDA（可选，用于 GPU 加速）
- MySQL 数据库

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置说明

编辑 `config.py` 文件，配置以下内容：

1. **数据库连接**：
```python
DB_CONFIG = {
    'host': 'your_host',
    'user': 'your_user',
    'password': 'your_password',
    'database': 'Tour',
    'charset': 'utf8mb4'
}
```

2. **模型路径**：
```python
MODEL_PATH = "/opt/model/gemma2"
EMBEDDING_MODEL_PATH = "/opt/model/bge-base-zh-v1.5"
```

3. **数据路径**：
```python
DATA_FILE_PATH = "./data/data.txt"
FAISS_INDEX_PATH = "./data/tour.faiss"
```

### 运行应用

```bash
streamlit run main.py
```

应用将在浏览器中自动打开，默认地址：`http://localhost:8501`

## 使用说明

1. **设置偏好**：在左侧边栏调整山川、海洋、平原的偏好分数
2. **输入问题**（可选）：在"初始问题"框中输入想了解的内容
3. **获取推荐**：点击"开始推荐"按钮，系统将推荐最匹配的目的地
4. **互动问答**：在聊天框中继续提问，了解更多目的地信息

## 推荐目的地

- **山西**：适合喜欢山川地貌的用户，拥有丰富的山地景观和历史遗迹
- **海南**：适合喜欢海洋的用户，热带海滩和水上活动丰富
- **北京**：适合喜欢平原景观的用户，历史文化名城，人文景点众多

## 工作原理

1. **推荐引擎**：根据三种地貌的评分，选择得分最高的对应目的地
2. **RAG 检索**：用户提问时，从知识库中检索相关景点信息
3. **LLM 生成**：结合检索结果和对话历史，生成个性化回答
4. **向量索引**：使用 FAISS 进行高效的语义相似度搜索

## 性能优化

- 模型使用 `@st.cache_resource` 缓存，避免重复加载
- FAISS 索引持久化到磁盘，加速启动
- 对话历史限制为最近 10 条消息，控制上下文长度
- 支持 GPU 加速推理

## 常见问题

**Q: 模型加载失败怎么办？**  
A: 检查模型路径是否正确，确保模型文件已下载到指定位置。

**Q: 数据库连接失败？**  
A: 检查 `config.py` 中的数据库配置，确保 MySQL 服务正常运行。

**Q: RAG 功能不可用？**  
A: 确保 `data/data.html` 文件存在，且嵌入模型已正确加载。

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，欢迎提交 Issue。
