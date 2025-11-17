import sqlalchemy
from sqlalchemy.sql import and_, or_, not_

en = sqlalchemy.create_engine("mysql://root:123456@localhost/testdb")

# 创建元数据
meta_data = sqlalchemy.MetaData()

# 创建表
person = sqlalchemy.Table(
    "person", meta_data,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String(32)),
    sqlalchemy.Column("age", sqlalchemy.Integer)
)
"""
name: 表名
meta_data: 元数据
Column: 字段信息
"""
# 创建表
meta_data.create_all(en)

print("_____插入_____")

# 插入一个记录
insert = person.insert().values(name="张三", age=18)
with en.connect() as con: # 创建连接
    result = con.execute(insert) # 执行插入
    print(result.inserted_primary_key) # 获取插入的id
    con.commit() # 提交

# 多条记录
insert = person.insert()
with en.connect() as con:
    result = con.execute(insert, [
        {"name": "张三", "age": 18},
        {"name": "张三", "age": 12},
        {"name": "张三", "age": 20}
    ])
    print(result.inserted_primary_key)
    con.commit()

print("_____查询_____")
# 普通查询
select = person.select()
with en.connect() as con:
    result = con.execute(select)
    for row in result:
        print(row) # 这是一个元组
    # 或者直接提取数据，结果集大不建议
    fetchResult = result.fetchall()
    print(fetchResult)
    # 取一个数据
    fetchOne = result.fetchone()
    print(fetchOne)

# 条件查询
select = person.select().where("张三" == person.c.name).where(person.c.age > 18) # 创建查询条件
with en.connect() as con:
    result = con.execute(select)
    for row in result:
        print(row)

# 带符号的条件的查询
select = person.select().where(and_(person.c.name == "张三", person.c.age > 18)) # 两个条件and
with en.connect() as con:
    result = con.execute(select)
    for row in result:
        print(row)

# 更新
# select = person.update().values(name="李四") # 全部全部为李四
select = person.update().where(1 == person.c.id).values(name="王五")
with en.connect() as con:
    result = con.execute(select)
    print(result.rowcount) # 影响的行数
    con.commit()

# 删除
select = person.delete().where(1 == person.c.id) # 删除id为1的
with en.connect() as con:
    result = con.execute(select)
    print(result.rowcount)
    con.commit()