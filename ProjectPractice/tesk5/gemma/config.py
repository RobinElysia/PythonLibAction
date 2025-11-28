"""配置文件"""

# 数据库链接设置
DB_CONFIG = {
    'host': '39.106.49.80',
    'user': 'root',
    'password': '000745psQ',
    'database': 'Tour',
    'charset': 'utf8mb4'
}

# 模型路径
MODEL_PATH = "/opt/model/gemma2"
EMBEDDING_MODEL_PATH = "/opt/model/bge-base-zh-v1.5"

# 数据路径
DATA_FILE_PATH = "./data/data.html"
FAISS_INDEX_PATH = "./data/tour.faiss"

# 推荐配置
RECOMMENDATION_MAP = {
    "山西": {
        "en": "Shanxi",
        "advice": "您似乎最喜欢<strong>山川地貌</strong>，<strong>山西</strong>拥有丰富的山地景观和历史遗迹，是您的理想选择。",
        "type": "mountain"
    },
    "海南": {
        "en": "Hainan",
        "advice": "您对<strong>海洋</strong>有浓厚的兴趣，<strong>海南</strong>以其热带海滩和水上活动而闻名，非常适合您。",
        "type": "ocean"
    },
    "北京": {
        "en": "Beijing",
        "advice": "您更倾向于<strong>平原景观</strong>，<strong>北京</strong>作为历史文化名城和首都，拥有众多人文景点，方便游览。",
        "type": "plain"
    }
}
