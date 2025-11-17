import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

mydb = myclient["test"] # 创建数据库
# 数据库只有在内容插入后才会创建! 就是说，数据库创建后要创建集合(数据表)并插入一个文档(记录)，数据库才会真正创建。

dblist = myclient.list_database_names() # 获取所有数据库名称
if "test" in dblist:
  print("数据库已存在！")

mydbMap = mydb["testMap"] # 创建集合
# 集合只有在内容插入后才会创建! 就是说，创建集合(数据表)后要再插入一个文档(记录)，集合才会真正创建。

collist = mydb.list_collection_names() # 获取所有集合名称
if "sites" in collist:   # 判断 sites 集合是否存在
  print("集合已存在！")

print("-----------------插入--------------------")

# 插入文档
myData = {
  "name": "John",
  "address": "Highway 37"
}
x = mydbMap.insert_one(myData)
print("文档插入成功，文档的 id 为：", x.inserted_id)

# 插入列表map
mylist = [
    {"name": "Taobao", "alexa": "100", "url": "https://www.taobao.com"},
    {"name": "QQ", "alexa": "101", "url": "https://www.qq.com"},
    {"name": "Facebook", "alexa": "10", "url": "https://www.facebook.com"},
    {"name": "知乎", "alexa": "103", "url": "https://www.zhihu.com"},
    {"name": "Github", "alexa": "109", "url": "https://www.github.com"}
]
x = mydbMap.insert_many(mylist)
print("列表插入成功，列表的 id 为：", x.inserted_ids)

# 自定义 id
mylist = [
    {"_id": 1, "name": "Taobao", "alexa": "100", "url": "https://www.taobao.com"},
    {"_id": 2, "name": "QQ", "alexa": "101", "url": "https://www.qq.com"},
    {"_id": 3, "name": "Facebook", "alexa": "10", "url": "https://www.facebook.com"},
]
x = mydbMap.insert_many(mylist)
print("列表插入成功，列表的 id 为：", x.inserted_ids)

print("-----------------查询--------------------")

# 查询第一个
result = mydbMap.find_one()
print(result)

# 查询所有
for x in mydbMap.find():
    print(x)

# 指定字段查询
# filter：查询条件，是一个文档。如果为空 {}，则匹配集合中的所有文档。
#
# projection：投影，指定返回文档中应包含或排除哪些字段。1 表示包含，0 表示排除。
# 投影文档不能混合使用包含和排除（_id 字段除外）
#
# options：其他选项，如排序、限制数量等。
for x in mydbMap.find({}, {"_id": 0, "name": 1, "address": 1}):
    print(x)

# 批次查询
for x in mydbMap.find().limit(2):
    print(x)

print("-----------------更新--------------------")
# filter: 查询条件，用于匹配要更新的文档
#
# update: 更新操作，指定如何修改文档
#
# options (可选): 额外选项，如 upsert
myquery = {"address": "Highway 37"}
values = {"$set": {"address": "Highway 37, New York"}}
mydbMap.update_one(myquery, values) # 更新第一个匹配的文档
print("文档更新成功")

print("-----------------删除--------------------")
# 删除单个
myquery = { "name": "Taobao" }
mydbMap.delete_one(myquery)

# 批量删除
# 删除以"Q"开头的
result1 = mydbMap.delete_many({"name": {"$regex": "^Q"}})

# 删除alexa 大于 80 的
result2 = mydbMap.delete_many({"alexa": {"$gt": 80}})

# $eq	等于	{"age": {"$eq": 25}}
# $ne	不等于	{"age": {"$ne": 25}}
# $gt	大于	{"age": {"$gt": 25}}
# $gte	大于等于	{"age": {"$gte": 25}}
# $lt	小于	{"age": {"$lt": 25}}
# $lte	小于等于	{"age": {"$lte": 25}}
# $in	在数组中	{"status": {"$in": ["active", "pending"]}}
# $nin	不在数组中	{"status": {"$nin": ["inactive", "deleted"]}}
#
# $and	逻辑与	{"$and": [{"age": {"$gt": 25}}, {"status": "active"}]}
# $or	逻辑或	{"$or": [{"age": {"$lt": 18}}, {"age": {"$gt": 65}}]}
# $not	逻辑非	{"age": {"$not": {"$lt": 18}}}
# $nor	逻辑或非	{"$nor": [{"price": 1.99}, {"sale": true}]}
#
# $exists	字段是否存在	{"email": {"$exists": true}}
# $type	字段类型匹配	{"age": {"$type": "int"}}
#
# $all	包含所有指定元素	{"tags": {"$all": ["mongodb", "python"]}}
# $elemMatch	数组元素匹配条件	{"results": {"$elemMatch": {"$gte": 80, "$lt": 90}}}
# $size	数组大小	{"tags": {"$size": 3}}
#
# $regex	正则表达式匹配	{"name": {"$regex": "^张"}}
# $text	文本搜索	{"$text": {"$search": "mongodb tutorial"}}
# $expr	聚合表达式	{"$expr": {"$gt": ["$price", "$discount"]}}
# $mod	取模运算	{"age": {"$mod": [2, 0]}}
# $jsonSchema	JSON模式匹配	{"$jsonSchema": {"bsonType": "object"}}