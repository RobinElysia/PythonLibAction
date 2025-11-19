from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def get_data():
    # 加载数据集
    data = pd.read_csv(r"D:\code\python\LearnPyLib\PytorchLib\utils\DataLoad\train.csv")
    print(data.shape)
    # 划分训练集和测试集
    X = data.drop("label", axis=1)
    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # 归一化
    preprocessor = MinMaxScaler()
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    return x_train, x_test, y_train.values, y_test.values