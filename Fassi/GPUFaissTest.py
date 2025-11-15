import faiss
import numpy as np

np.random.seed(0)

def test01():
    data = np.random.rand(10000, 256)
    # 创建资源
    res = faiss.StandardGpuResources()
    # 在 CPU 创建索引
    index_cpu = faiss.IndexFlatL2(256)
    print(index_cpu)
    # 将索引转到 GPU
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    """
    参数1：GPU 使用资源
    参数2：GPU 设备编号
    参数3：转移的索引
    """
    print(index_gpu)
    # 插入数据
    index_gpu.add(data)
    # 向量搜索
    D, I = index_gpu.search(np.random.rand(2, 256), k=2)
    print(D)
    print(I)

if __name__ == "__main__":
    test01()