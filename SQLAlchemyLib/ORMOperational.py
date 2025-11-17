from typing import Type, List
from sqlalchemy import Engine
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

    # 插入数据
    Base.metadata.create_all(en)