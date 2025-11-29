"""数据库工具函数"""
import pymysql
import random
import streamlit as st
from config import DB_CONFIG


def get_db_connection():
    """获取数据库链接"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        st.error(f"数据库连接失败，因为: {e}")
        return None


def get_destination_info(destination_en_name):
    """获取目的地图片路径"""
    query = "SELECT pic FROM destination WHERE city = %s"

    try:
        conn = get_db_connection()
        if conn is None:
            return None
        with conn.cursor() as cursor:
            cursor.execute(query, (destination_en_name,))
            results = cursor.fetchall()
        conn.close()

    except Exception as e:
        return None

    if results:
        return random.choice(results)[0]

    return None
