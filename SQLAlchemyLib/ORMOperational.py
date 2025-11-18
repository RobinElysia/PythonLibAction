from typing import Type, List
from sqlalchemy import Engine, insert, select
from sqlalchemy.sql import and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

import sqlalchemy

def create_en() -> tuple[Engine, Type[DeclarativeBase], sessionmaker[Session]]:
    en = sqlalchemy.create_engine("mysql://root:123456@localhost/testdb")

    # 创建基类
    Base = declarative_base()

    # 创建会话
    Session = sessionmaker(bind=en)
    return en, Base, Session

# 创建连接
en, Base, Session = create_en()

class Person(Base):
    __tablename__ = "person"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    name = sqlalchemy.Column(sqlalchemy.String(32))
    age = sqlalchemy.Column(sqlalchemy.Integer)

def insert_data(person: Person, session: sessionmaker[Session], Base: Type[DeclarativeBase]):
    # 创建会话
    insert_session = session()
    # 添加数据
    insert_session.add(person)
    # 提交数据
    insert_session.commit()

def insert_M_data(person: List[Person], session: sessionmaker[Session], Base: Type[DeclarativeBase]):
    # 创建会话
    insert_session = session()
    # 批量添加数据
    insert_session.add_all(person)
    # 提交数据
    insert_session.commit()

def select_data(person: Person, en: Engine, session: sessionmaker[Session], Base: Type[DeclarativeBase]):
    # 创建会话
    select_session = session()
    # 查询所有
    result = select_session.query(person).all()
    for row in result:
        print(row)

def select_con_data(person: Person, session: sessionmaker[Session], Base: Type[DeclarativeBase]):
    # 创建会话
    select_session = session()
    # 条件查询第一个，查询可以为空
    result = select_session.query(person).filter(and_(person.name == "张三")).first()
    # 结果集只有一条记录使用one，查询不能为空
    # result = select_session.query(person).filter(and_(person.name == "张三")).one()
    # 结果集只有一条记录，但是可以为空
    # result = select_session.query(person).filter(and_(person.name == "张三")).scalar()
    print(result)

def update_base_query_data(person: Person, session: sessionmaker[Session], Base: Type[DeclarativeBase]):
    # 创建会话
    update_session = session()
    # 更新
    result = update_session.query(person).one()
    # 修改
    result.name = "赵四"
    # 提交数据
    update_session.commit()

def update_func_data(person: Person, session: sessionmaker[Session], Base: Type[DeclarativeBase]):
    # 创建会话
    update_session = session()
    # 更新
    result = update_session.query(person).update({"name": "小潘"})
    # 提交数据
    update_session.commit()

# 批量插入
def insert_M_useValue_data(session: sessionmaker[Session]):
    # 创建会话
    insert_session = session()
    # 批量添加数据
    insert_session.execute(
        insert(Person).values([
            {"id": 4, "name": "王五", "age": 18},
            {"id": 5, "name": "赵刘", "age": 18},
        ])
    )
    # 提交数据
    insert_session.commit()

# 嵌套查询的批量插入
def insert_M_useSelect_data(session: sessionmaker[Session]):
    # 创建会话
    insert_session = session()
    insert_session.execute(
        insert(Person).values(
            [
                {
                    "id": 6, "name": "bob", "age": select(Person.age).where(Person.id == 1)
                },
                {
                    "id": 7, "name": "lili", "age": select(Person.age).where(Person.id == 2)
                },
            ]
        )
    )# 查询id为1 2的年龄作为6 7年龄

# 更新删除同理也可以使用execute进行批量

# 什么？你问我事务和多数据源？事务天然支持，多数据源一行代码搞定
# with Session(engine) as session1, session1.begin(), Session(engine2) as session2.begin():

if __name__ == "__main__":
    # 插入数据
    person = Person(id=1, name="张三", age=18)
    insert_data(person, Session, Base)

    # 批量插入
    ps = [
        Person(id=2, name="王五", age=18),
        Person(id=3, name="李四", age=18),
    ]
    insert_M_data(ps, Session, Base)

    # 查询
    person = Person(id=1)
    select_data(person, en, Session, Base)

    # 条件查询
    person = Person(id=1)
    select_con_data(person, Session, Base)

    # 修改1
    person = Person(id=1)
    update_base_query_data(person, Session, Base)

    # 修改2
    person = Person(id=1)
    update_func_data(person, Session, Base)

    # 批量插入
    insert_M_useValue_data(Session)

    # 嵌套查询的批量插入
    insert_M_useSelect_data(Session)

    # 插入数据
    Base.metadata.create_all(en)