import pandas as pd

# 简单的时间
d = pd.Timestamp("2023-01-01 10:22")
d1 = pd.Timestamp("2023-01-01 12:22:00") # 和d是同一天
print(d)
print(d.year, d.month, d.day, d.hour, d.minute, d.second) # 打印时间
print(d.quarter) # 获取季度
print(d.weekday()) # 获取星期几
print(d.day_name()) # 获取星期几
print(d.days_in_month) # 获取月份天数
# 判断
print("————判断————")
print(d.is_month_start) # 判断是否是月份开始
print(d.is_month_end) # 判断是否是月份结束
print(d.is_quarter_start) # 判断是否是季度开始
print(d.is_quarter_end) # 是否是季度结束
print(d.is_year_start) # 是否是年份开始
print(d.is_year_end) # 是否是年份结束
print(d.is_leap_year) # 是否是闰年

# 转换
print("————转换————")
print(d.to_pydatetime()) # 转换为python时间
print(d.to_datetime64()) # 转换为numpy时间，可以直接作用于DataFrame
# 类似于：df["datatime"].dt.day_time()，获取时间属性并转化为星期
# 因为获取的是Series对象，我们需要使用 dt 访问器转化为每个时间，才能对其进行day_name()
print(d.to_numpy()) # 转换为numpy时间
print(d.to_julian_date()) # 获取儒略日
print(d.to_period("D"))
print(d1.to_period("D")) # 和d是同一天
# D , M , Y , W , Q

# 读取是文件时候直接指定时间解析
df = pd.read_csv("complex_sales_data.csv", parse_dates=["交易日期"]).head(10) # 将date解析为时间类型
print(df.dtypes) # 查看交易日期的类型
# 设置为索引
df.set_index("交易日期", inplace=True)
# 排序
df = df.sort_index()
# 排序后才可以切片
print(df.loc["2023-01-13":"2023-03-13"])

# 时间间隔
print("————时间间隔————")
d2 = pd.Timestamp("2023-01-01 10:22")
d3 = pd.Timestamp("2023-01-01 12:22:00")
t = d3 - d2
print(t)
print(type(t)) # Timedelta类型

# 重新采样
print("————重新采样————")
df.dropna(inplace=True, subset=["利润", "客户评分"])
print(df)
print(df[["客户评分", "利润"]].resample("QE").mean()) # 季度平均重采样