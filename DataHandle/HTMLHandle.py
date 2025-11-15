from bs4 import BeautifulSoup

with open("test.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "lxml")
"""
使用BS创建对象
参数一：可以是html的字符串，或者文件对象
参数二：指定解析器，或者不指定使用默认
"""
print("________基本调用__________")
print(soup.title) # 获取第一个title节点
print(soup.meta) # 获取第一个meta节点

print("________三个重要信息__________")

# 获取标签的三个重要信息
# 内容
print(soup.a.string)
# 属性
print(soup.a.attrs)
# tag名称
print(soup.a.name)

print("________子标签调用__________")
# tag.tag
# 内容
print(soup.li.a.string)
# 属性
print(soup.li.a.attrs)
# tag名称
print(soup.li.a.name)

print("________关联选择__________")
# 子节点
print(soup.li.contents) # 返回列表
print(soup.li.children) # 返回迭代器
for i in soup.head.children:
    print("子节点", i)

# 父节点
print("父节点" , soup.meta.parent)

# 祖先节点
print(soup.a.parents) # 生成器
print("祖先节点:" , list(enumerate(soup.ul.parents)))

# 兄弟节点
print("兄弟节点:" , soup.meta.next_sibling) # 下一个兄弟节点
print("兄弟节点:" , list(
    enumerate(soup.meta.next_siblings))) # 下一个兄弟节点生成器
print("兄弟节点:" , soup.title.previous_sibling) # 上一个兄弟节点
print("兄弟节点:" , list(
    enumerate(soup.title.previous_siblings))) # 上一个兄弟节点生成器

print("________css选择器__________")
print(soup.select("section label")) # 标签选择器，html44、47行
print(soup.select("#section3 .aaa")) # id + class选择器，41行
print(soup.select("#section3 #name")) # 标签id选择器，45行

print("________css选择器高级使用方法__________")
for i in soup.select("form"):
    print(i.select("input")[0].attrs) # input标签，html45行
    print(i.select("label")[0].string) # 44行
    # i.select("input")[0]、i.select("label")[0]是一个tag

import re
print("________方法选择器__________")
print("获取所有a标签", soup.find_all(name="a"))
print("获取id为name的标签", soup.find_all(id="name"))
print("获取class为aaa的标签", soup.find_all(class_="aaa"))
print("属性值查询，获取class为aaa的标签", soup.find_all(attrs={"class": "aaa"}))
print("获取文本为 章节1 的标签", soup.find_all(string="章节1"))
print("获取以章节开头的标签，正则匹配", soup.find_all(name=re.compile('^in')))
print("获取以章节开头的标签，正则匹配，限制数量", soup.find_all(name=re.compile('^in'), limit=1))
print("获取以章节为开头的", soup.find_all(string=re.compile("^章节")))
print("获取所有标签", soup.find_all(True))

# find 符合条件的第一个元素，使用上除了limit，其他与find_all一样
print("find", soup.find(name="a"))
print("find", soup.find(id="name"))
print("find", soup.find(class_="aaa"))
print("find", soup.find(attrs={"class": "aaa"}))
print("find", soup.find(string="章节1"))