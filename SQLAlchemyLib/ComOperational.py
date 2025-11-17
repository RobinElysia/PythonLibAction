from typing import Tuple

import sqlalchemy
from sqlalchemy.sql import and_

def create_en() -> Tuple[sqlalchemy.engine.base.Engine, sqlalchemy.MetaData]:
    en = sqlalchemy.create_engine("mysql://root:123456@localhost/testdb")

    # 创建元数据
    meta_data = sqlalchemy.MetaData()
    return en, meta_data

def create_tb(meta_data) -> Tuple[sqlalchemy.Table, sqlalchemy.Table]:
    # 部门
    department = sqlalchemy.Table(
        "department", meta_data,
        sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column("name", sqlalchemy.String(32))
    )

    # 员工
    employee = sqlalchemy.Table(
        "employee", meta_data,
        sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column("name", sqlalchemy.String(32)),
        sqlalchemy.Column("age", sqlalchemy.Integer),
        # 外键
        sqlalchemy.Column("department_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("department.id"))
    )
    return department, employee

def insert_data(en, department, employee):
    with en.connect() as con:
        con.execute(department.insert(), [
            {"name": "it"},
            {"name": "hr"}
        ])
        con.execute(employee.insert(), [
            {"name": "张三", "age": 18, "department_id": 1},
            {"name": "李四", "age": 12, "department_id": 1},
            {"name": "王五", "age": 20, "department_id": 2}
        ])
        con.commit()

def select_data(en, department, employee):
    with en.connect() as con:
        # 联合查询
        join = employee.join(department, department.c.id == employee.c.department_id)
        # q = sqlalchemy.select(join).where(and_(department.c.name == "it")) # 创建查询条件
        q = sqlalchemy.select(employee).select_from(join).where(and_(department.c.name == "it")) # 不带部门
        # 执行查询
        result = con.execute(q)
        for row in result:
            print(row)

if __name__ == "__main__":
    # 创建连接和元数据
    en, meta_data = create_en()

    # 创建表
    department, employee = create_tb(meta_data)

    # 查询
    select_data(en, department, employee)

    # 插入数据
    meta_data.create_all(en)