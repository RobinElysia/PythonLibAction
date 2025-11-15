import faiss
import numpy as np

np.random.seed(0)

def test01():
    data = np.random.rand(10000, 256)
    index = faiss.IndexFlatL2(256)
    index = faiss.IndexIVFPQ(index, 256, 100, 32, 10)
    """
    1. 基本索引
    2. 维度
    3. 质心数
    4. 桶数
    5. 单向量划分数
    
    比如：
    数据矩阵长度256，个数10000，那么桶数就是划分10000，而单向量划分数就是划分长度256
    一个向量划分为 n 份，n 越大，精度越高。划分最后是使用 8 位的数代替 256 精度
    """
    # 训练质心
    index.train(data) # 训练IVF索引
    index.add(data) # 添加数据
    index.add_with_ids(data, np.arange(10000)) # 添加数据，索引ID从0开始
    print(index.ntotal)
    # 搜索
    query = np.random.rand(1, 256)
    index.nprobe = 3  # 设置搜索的质心数
    index.search(query, 5)
    I, D = index.search(query, 5)
    print(I)
    print(D)

    faiss.write_index(index, "IndexIVFPQ.faiss")

if __name__ == "__main__":
    test01()