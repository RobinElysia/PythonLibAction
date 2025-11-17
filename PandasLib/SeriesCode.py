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
print("索引值", s5.keys()) # 等价于 s5.index
print("索引值", s5.index)
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
