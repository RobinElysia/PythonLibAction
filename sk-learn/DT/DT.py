from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree import ExtraTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
import graphviz

"""
树算法
__all__ = [
    "BaseDecisionTree",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
    "export_graphviz",
    "export_text",
    "plot_tree",
]

一般信息熵计算要比基尼系数复杂，并且信息熵会在噪声多的数据中出现过拟合，要根据情况而定
"""
def DT_Classifier():
    wine = load_wine()  # 获取数据
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)  # 划分数据集
    clf = DecisionTreeClassifier(
        criterion="entropy",
        random_state=0,  # 设置随机数种子，保证每次运行结果一致
        splitter="best",  # 选择最佳特征，默认best, 或者使用random
    )  # 创建分类树
    clf.fit(X_train, y_train)  # 训练
    print(clf.score(X_test, y_test))  # 测试
    dot_data = export_graphviz(clf, out_file=None)  # 创建dot文件
    graph = graphviz.Source(dot_data)  # 绘制
    print(graph)
    # 查看节点贡献
    print([*zip(("酒精", "苹果酸", "灰"
                 , "灰的碱性", "镁", "总酚"
                 , "类黄酮", "非黄烷类酚类", "花青素",
                 "颜色强度", "色调", "马士特罗",
                 "脯氨酸"), clf.feature_importances_)])
    # 剪纸操作：
    # 最大深度（常用，建议从三开始试）max_depth
    # 最小叶子样本数：min_samples_leaf，太小会过拟合，太大会欠拟合（建议从5开始）
    # 最少样本可分：min_samples_split，
    # 最大特征：max_features，建议使用PCA进行降维，而不是使用这个参数
    # 最小信息增益可划分：min_impurity_decrease

    # 权重：
    # 当存在分类偏移时，比如一个类别的样本数量占比极低，此时就需要施加权重
    # 类别权重：class_weight，默认为None，决策树会自动调整，但是如果自动调整不行，那就手动设置参数
    # 同时需要剪枝操作：min_fraction_leaf

    # 获取节点和预测
    print(clf.apply(X_test), clf.predict(X_test))

def DT_Regressor():
    # 参数和分类树一摸一样，但是没有离散值的分类样本不均很问题
    # 使用L2、L1和费尔德曼距作为损失函数（代替信息增益和基尼指数）
    # score返回的是R2，不是MSE
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)
    clf = DecisionTreeRegressor(random_state=0)
    # 交叉检验
    print(cross_val_score(clf, wine.data, wine.target, cv=5))
    print(cross_validate(clf, wine.data, wine.target, cv=5)) # 通过交叉验证评估指标，并记录Fitscore时间。
    pass

if __name__ == '__main__':
    DT_Classifier()