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