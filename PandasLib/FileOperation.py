import pandas as pd
import json

df_csv = pd.read_csv("complex_sales_data.csv")
print(df_csv.head())
print(df_csv.tail())

df_csv.head().to_csv("new.csv") # 创建一个新文件，保存前五个

print("--------JSON读取--------")
"""
df_json = pd.read_json("complex_sales_data_json.json")
print(df_json.head())
"""

# 也可以这么读取JSON
with open("complex_sales_data_json.json", "r", encoding="utf-8") as f:
    data = json.load(f)
df_json = pd.DataFrame(data)
print(df_json.head())

# 保存为CSV文件
df_json.to_json("new_json.json", orient="records", indent=4, force_ascii=False)
# 保存为JSON文件,  orient="records" 表示将DataFrame转换为JSON列表, indent=4 表示缩进4个空格,
# force_ascii=False 表示将所有非ASCII字符转换为Unicode转义序列