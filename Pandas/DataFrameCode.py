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