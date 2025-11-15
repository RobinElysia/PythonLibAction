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
df.to_json("complex_sales_data_json.json", orient="records", indent=4, force_ascii=False)
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