import pandas as pd

# 读取文件
df = pd.read_csv("complex_sales_data.csv", parse_dates=["交易日期"]).head(10)
# 缺失值统计
print(df.isnull().sum())
print(df.dtypes)
# 分组
df_group = df.groupby("地区")
print(df_group.groups) # 获取分组
# 或者查看某个具体的分组（按照城市分组）
df_group_data = df_group.get_group("Central") # 拿到某个城市的数据
print(type(df_group_data)) # 这是一个DataFrame
print("这是利润\n", df_group_data["利润"]) # 获取利润
print("这是利润平均值", df_group_data["利润"].mean()) # 获取平均值
print("这是利润平均值", df_group_data["利润"].mean().round(2)) # 取两位小数

# 多条件分组
print("_____多条件分组_____")
# 拿到(Central, Completed)两个共同条件的数据，这是一个元组
df_M_group = df.groupby(["地区", "交易状态"]).get_group(("Central", "Completed"))
print(type(df_M_group)) # 这是一个DataFrame
print(df_M_group[["利润"]]) # 注意[["利润"]]还是一个DataFrame

# 其他操作同理