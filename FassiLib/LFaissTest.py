import faiss
import numpy as np

# 简单索引
np.random.seed(0) # 设置随机数种子

def simple_index():
    """
    简单的线性索引
    """
    data = np.random.rand(10000, 256) # 随机生成10000行256列的数据
    # 参数为索引维度
    index = faiss.IndexFlatL2(256) # L2范数，欧式索引
    """
    index = faiss.IndexFlatIP(index) # 内积索引
    index = faiss.index_factory(256, "Flat", faiss.METRIC_INNER_PRODUCT) # 创建内积索引，等效于11行
    index = faiss.index_factory(256, "Flat", faiss.METRIC_L2) # 创建L2范数索引，等效于10行
    """

    index.add(data) # 添加数据

    print(index.ntotal) # 索引的样本数量
    print(index.d) # 索引的维度

    # 创建新数据
    query = np.random.rand(1, 256)
    # 搜索近似的5个数据
    index.search(query, 5)
    # 解构数据
    I, D = index.search(query, 5)
    print(I)
    print(D)

    # 存储索引
    faiss.write_index(index, "IndexFlatL2.faiss")

    # 删除数据
    index.remove_ids(np.array([0,1,2,3]))
    # 查看数据个数
    print(index.ntotal)

    # 删除所有向量数据
    index.reset()
    print(index.ntotal)

if __name__ == "__main__":
    simple_index()