import pandas as pd

df = pd.DataFrame(
    {
        '姓名': ['张三', '李四', '王五', '张三', '赵六', '李四', '孙七', '王五'],
        '年龄': [25, 30, 28, 25, 35, 30, 27, 28],
        '城市': ['北京', '上海', '广州', '北京', '深圳', '上海', '杭州', '广州'],
        '分数': [85, 92, 78, 85, 95, 92, 88, 78],
        '部门': ['技术部', '销售部', '市场部', '技术部', '人事部', '销售部', '研发部', '市场部']
    }
)
print(df)
print(df.duplicated()) # 判断是否有重复行
print(df.drop_duplicates()) # 删除重复行
# 根据名称去重
print(df.drop_duplicates(subset=['姓名']))
# 保存最新的数据去重
print(df.drop_duplicates(keep="last"))

# 数据类型的转换
print(df.dtypes)
df[["分数"]] = df[["分数"]].astype("int32")
print("转换后的：\n", df.dtypes)
df[["部门"]] = df[["部门"]].astype("category") # 转换为分类型
print("转换后的：\n", df.dtypes)