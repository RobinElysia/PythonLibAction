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
