import pymysql
from pymysql.connections import Connection

def main() -> Connection:
    """
        创建连接对象
    """
    # Connect to the database
    con = pymysql.connect(
        host='localhost',
        user='root',
        password='',
        charset='utf8',
        db='test'
    )

    # 创建游标对象
    return con

if __name__ == "__main__":
    con = main() # 获取连接对象
    cu = con.cursor()

    try:
        cu.execute("CREATE DATABASE IF NOT EXISTS aaa")
        con.commit() # 提交创建，新增、修改、删除必须提交
        print(cu.fetchall()) # 获取结果
    except:
        con.rollback() # 事务回滚

    cu.execute("SHOW TABLES")
    print(cu.fetchone())  # 一行一行获取结果
    print(cu.fetchmany(2))  # 获取 2 行结果
    print(cu.fetchall()) # 获取结果，这里游标已经没有结果了，所以只能获取一次

    cu.execute(
        "SELECT * FROM user WHERE name=%s and password=%s",('admin', 'admin')
        )# 占位符，防止sql注入
    print(cu.fetchone())
    print(cu.fetchall())

    # cu.execute(
    #         "SELECT * FROM user WHERE name=%(name)s and password=%(password)s",{
    #             "name" : name,
    #             "password" : password
    #         }
    #     )# 指名占位符，防止sql注入

    cu.close() # 关闭游标
    con.close() # 关闭连接
