import evaluate
from evaluate.visualization import radar_plot # 可视化雷达图

def Easy_Evaluate():
    """
    简单介绍
    :return:
    """
    # 查看评估工具包支持的模型
    for i in evaluate.list_evaluation_modules():
        print(i)
    # 不想要社区实现的模型评估
    # print(evaluate.list_evaluation_modules(include_community=False))
    # 查看细节
    # print(evaluate.list_evaluation_modules(with_details=True))

    # 加载评估函数
    accuracy = evaluate.load("accuracy") # 在上述打印中有 accuracy
    # 查看函数说明
    print(accuracy.description)
    """
    Accuracy is the proportion of correct predictions among the total number of cases processed. It can be computed with:
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
     Where:
    TP: True positive
    TN: True negative
    FP: False positive
    FN: False negative
    """
    print(accuracy.inputs_description) # 输入参数说明
    """
    Examples:
        Example 1-A simple example
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
            >>> print(results)
            {'accuracy': 0.5}
    
        Example 2-The same as Example 1, except with `normalize` set to `False`.
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], normalize=False)
            >>> print(results)
            {'accuracy': 3.0}
    
        Example 3-The same as Example 1, except with `sample_weight` set.
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4])
            >>> print(results)
            {'accuracy': 0.8778625954198473}
    """
    # 或者直接全部打印
    print(accuracy)

def Algorithm_Evaluate():
    """
    算法评估
    :return:
    """
    # 加载评估函数
    accuracy = evaluate.load("accuracy")
    print("简单的模型评估：", accuracy.compute(
        references=[0, 1, 2, 0, 1, 2],
        predictions=[0, 1, 1, 2, 1, 0],
        sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4]
    )) # 输出计算评估结果，references是预测结果，predictions是真实结果，sample_weight是权重

    # 或者，迭代计算
    for ref, pred in zip([0, 1, 2, 0, 1, 2], [0, 1, 1, 2, 1, 0]):
        accuracy.add(references=ref, predictions=pred)
    print("迭代评估计算：", accuracy.compute())

    # 或者，使用add_batch方法
    for ref, pred in zip([[0,0],[1,1]], [[2,1],[0,2]]):
        accuracy.add_batch(references=ref, predictions=pred) # 不能使用sample_weight
    print("批量评估计算：", accuracy.compute())

def Mult_Algorithm_Evaluate():
    """
    多个评估指标计算
    :return:
    """
    # 加载多个评估函数
    metric = evaluate.combine(evaluations=["accuracy", "precision", "recall", "f1"])
    print(metric)
    # precision_score / recall_score / f1_score 等默认在二分类场景下只返回“正类”的分数，一旦类别数 >2 就不知道该返回哪一类了，于是抛出 ValueError。

    # 简单计算
    print(metric.compute(references=[1, 1, 1, 0, 1, 0],
        predictions=[0, 1, 1, 0, 1, 1]))

def Vision_Evaluate():
    """
    可视化
    """
    data = [
        {'accuracy': 0.95, 'precision': 1.00, 'recall': 0.55, 'f1': 0.71},
        {'accuracy': 0.60, 'precision': 0.65, 'recall': 0.95, 'f1': 0.78},
        {'accuracy': 0.75, 'precision': 0.70, 'recall': 0.60, 'f1': 0.65},
        {'accuracy': 0.50, 'precision': 0.55, 'recall': 0.50, 'f1': 0.52}
    ] # 评估结果
    models = ['model_1', 'model_2', 'model_3', 'model_4'] # 模型名称
    plot = radar_plot(data, models) # 绘制雷达图
    plot.show()

if __name__ == '__main__':
    """
    Transformers模型评估工具包
    """
    # Easy_Evaluate()
    # Algorithm_Evaluate()
    # Mult_Algorithm_Evaluate()
    Vision_Evaluate()