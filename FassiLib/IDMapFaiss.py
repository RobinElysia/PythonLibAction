import faiss
import numpy as np

np.random.seed(0)

def test01():
    """
    键映射
    """
    index_data = np.random.rand(1000000, 256)
    index = faiss.IndexFlatL2(256) # 创建索引对象
    index = faiss.IndexIDMap(index) # 升级为键映射索引对象
    index.add_with_ids(index_data, np.arange(1000000,2000000))  # 添加数据，索引ID从1000000开始
    print(index.ntotal) # 索引的样本数量
    pass

if __name__ == "__main__":
    test01()