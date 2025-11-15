import xml.etree.ElementTree as et
import xml.dom.minidom as Doc

# 解析xml文件
root = et.ElementTree(file="test.xml") # 创建ElementTree对象
print("对象：",root)
tree = root.getroot() # 获取根节点
print("根节点对象：", tree)
print("根节点标签：", tree.tag) # 获取根节点标签，就是最外层的标签
print("根节点属性：", tree.attrib) # 获取根节点属性

# 遍历获取子元素
for child in tree:
    # print("子元素标签：", child.tag,"子元素属性：", child.attrib)
    print(child[0].text)
    print(child[1].text)
    print(child[2].text)
    for grandchild in child:
         print("孙元素标签：", grandchild.tag, "孙元素文本", grandchild.text)

# 索引获取孙元素
print(tree[0][0].text)

# 查询元素
print(tree.find("company")) # 查询子元素
print(tree.find("person").find("name").text) # 查询首次出现的孙元素
print(tree.findall("person")[1].find("name").text) # 查询第二个孙元素

print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
# 创建xml文件
doc = Doc.Document() # 创建Document对象
root = doc.createElement("root") # 创建根节点
doc.appendChild(root) # 添加根节点

head = doc.createElement("head") # 创建子节点
root.appendChild(head) # 添加子节点

ch = doc.createElement("ch") # 创建孙节点
head.appendChild(ch) # 添加孙节点

text = doc.createTextNode("hello world")
ch.appendChild(text) # 添加文本节点

print(doc.toxml()) # 输出xml文件

# 保存xml文件
# with open("OutTest.xml", "w+") as f:
#     f.write(
#         doc.toprettyxml(encoding="utf-8")
#         .decode("utf-8")
#     )# 保存 xml 文件

print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
par = et.parse("test.xml")
print(par)
# 剩下的和解析xml文件一样，par 是一个 ElementTree 对象
