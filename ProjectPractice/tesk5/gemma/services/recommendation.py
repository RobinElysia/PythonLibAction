"""推荐服务"""
from config import RECOMMENDATION_MAP
from utils.db_utils import get_destination_info
from utils.image_utils import image_file_to_base64


def recommend_destination(mountain_score: float, ocean_score: float, plain_score: float):
    """
    根据用户对山地、海洋、平原的喜好分数，推荐最匹配的国内目的地。

    return:
    best_cn_name : str      中文目的地名
    advice       : str      推荐理由（含 HTML 格式）
    image_base64 : str      图片的 base64 编码
    mime_type    : str      图片 MIME 类型，如 'image/jpeg'
    """
    # 构建评分映射
    scores = {
        "山西": mountain_score,
        "海南": ocean_score,
        "北京": plain_score
    }
    
    # 找出得分最高的目的地
    best_cn_name = max(scores, key=scores.get)
    info = RECOMMENDATION_MAP[best_cn_name]

    # 查询数据库并读取图片
    image_path = get_destination_info(info["en"])
    image_base64, mime_type = image_file_to_base64(image_path)

    return best_cn_name, info["advice"], image_base64, mime_type
