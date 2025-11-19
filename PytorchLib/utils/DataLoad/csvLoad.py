from typing import Any

import pandas as pd
from numpy import ndarray


def csvLoad(file_path) -> tuple[ndarray, ndarray]:
    """
    csv文件读取
    :param file_path: 文件路径
    :return:
    """
    # 读取csv文件
    df = pd.read_csv(file_path)
    # 获取列名
    columns = df.columns().values # 转换为ndarray
    # 获取数据
    data = df.values
    return columns, data