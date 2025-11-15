import pandas as pd

df = pd.DataFrame(
    {
        'ID': [1, 2, 3],
        'Name': ['Alice Elysia', 'Bob Elysia', 'Charlie Elysia'],
        'Math': [85, 92, 78],
        'Science': [88, 90, 85],
        'English': [92, 85, 88]
    }
)
# df3["Name"]：返回 Series，适用于对单列进行数值计算、字符串操作等
#
# df3[["Name"]]：返回 DataFrame，适用于需要保持表格结构、选择多列或进行复杂数据处理的情况

# 转置
print(df.T)
# 变换列，宽转窄
df2 = pd.melt(df, id_vars=["ID", "Name"], var_name="科目", value_name="分数").sort_values("Name")
print(df2)
# 宽转高
df3 = df2.pivot(index=["ID", "Name"], columns="科目", values="分数")
print(df3)
print("___________________")
# 分列
df[["Frist Name","Last Name"]] = df["Name"].str.split(" ", expand=True)
# 以空格分裂名称，expand=True表示返回DataFrame
# df["Name"].str.split(" ", expand=True)，Series转str转DataFrame，赋值给原来的DataFrame
print(df)