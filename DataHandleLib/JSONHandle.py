import json as json


# json转字典
with open("test.json", "r", encoding="utf-8") as f:
    data = json.load(f) # 读取 json 文件

name = data["name"]
hobbies: list = data["hobbies"]
email = data["contact"]["email"]

print(name)
for i, hobby in enumerate(hobbies):
    print(i, hobby)
print(email)

# 字典转json
DataMap = json.dumps(data)
print(DataMap)
with open("OutTest.json", "w", encoding="utf-8") as f:
    json.dump(
        data, # 输入字典
        f, # 输出文件
        ensure_ascii=False, # 将所有非ASCII字符转换为Unicode转义序列
        indent=4, # 缩进
        sort_keys=False # 保持键的原始插入顺序，非排序
    )