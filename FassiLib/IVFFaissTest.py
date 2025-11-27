import faiss
import numpy as np

np.random.seed(0) # 设置随机数种子

def test01():
    """
    聚类倒排索引
    """
    index_data = np.random.rand(10000, 256) # 随机生成1000000行256列的数据
    qu = faiss.IndexFlatL2(256) # 创建索引对象
    index = faiss.IndexIVFFlat(qu, 256, 100) # 升级为 IVF PQ 索引对象
    """
    第一个参数：基本索引对象
    第二个参数：维度
    第三个参数：聚类的质心
    """
    index.train(index_data) # 训练IVF索引
    index.add(index_data) # 添加数据到索引对象
    faiss.write_index(index, "IndexIVFFlat.faiss")
    print(index.ntotal)

    query = np.random.rand(1, 256) # 创建查询数据
    I, D = index.search(query, 5) # 搜索近似的 5个数据
    print(I)
    print(D)

def test02():
    """
    简单的线性索引
    """
    data = np.random.rand(10000, 256)  # 随机生成10000行256列的数据
    # 参数为索引维度
    index = faiss.IndexFlatL2(256)  # L2范数，欧式索引

    index.add(data)  # 添加数据

    print(index.ntotal)  # 索引的样本数量
    print(index.d)  # 索引的维度

    # 创建新数据
    query = np.random.rand(1, 256)
    # 搜索近似的5个数据
    index.search(query, 5)
    # 解构数据
    I, D = index.search(query, 5)
    print(I)
    print(D)


if __name__ == '__main__':
    test01()
    # test02()