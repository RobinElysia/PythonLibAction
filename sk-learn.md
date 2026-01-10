## 分类（Classification）补充 

### 1. 线性模型补充

`SGDClassifier` - 随机梯度下降分类器

```python
from sklearn.linear_model import SGDClassifier

# 常用参数
model = SGDClassifier(
    loss='hinge',           # 损失函数：'hinge'(SVM), 'log'(逻辑回归), 'modified_huber'
    penalty='l2',           # 正则化：'l1', 'l2', 'elasticnet'
    alpha=0.0001,           # 正则化强度
    l1_ratio=0.15,          # ElasticNet混合比例(0-1)
    max_iter=1000,          # 最大迭代次数
    tol=1e-3,               # 停止容差
    learning_rate='optimal',# 学习率：'constant','optimal','invscaling','adaptive'
    eta0=0.0,               # 初始学习率
    random_state=42,
    n_jobs=-1               # 并行作业数
)
```

**适用场景**：大规模数据集、在线学习、稀疏数据

`Perceptron` - 感知器

```python
from sklearn.linear_model import Perceptron

model = Perceptron(
    penalty=None,           # 正则化：'l1','l2','elasticnet'
    alpha=0.0001,
    fit_intercept=True,     # 是否拟合截距
    max_iter=1000,
    tol=1e-3,
    shuffle=True,           # 每轮迭代洗牌
    random_state=42,
    n_jobs=-1
)
```

**特点**：最简单的线性分类器，适合大规模线性可分数据

`PassiveAggressiveClassifier` - 被动攻击分类器

```python
from sklearn.linear_model import PassiveAggressiveClassifier

model = PassiveAggressiveClassifier(
    C=1.0,                  # 正则化强度
    fit_intercept=True,
    max_iter=1000,
    tol=1e-3,
    early_stopping=False,   # 提前停止
    validation_fraction=0.1,# 验证集比例
    n_iter_no_change=5,     # 无改进迭代次数
    random_state=42,
    n_jobs=-1
)
```

**特点**：在线学习，对新样本快速适应，适合流数据

### 2. 朴素贝叶斯补充

`GaussianNB` - 高斯朴素贝叶斯

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB(
    var_smoothing=1e-9      # 方差平滑，防止零方差
)
```

**适用场景**：连续特征，假设特征服从正态分布

`MultinomialNB` - 多项朴素贝叶斯

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(
    alpha=1.0,              # 拉普拉斯平滑参数
    fit_prior=True,         # 是否学习先验概率
    class_prior=None        # 手动指定先验概率
)
```

**适用场景**：离散特征计数（如文本词频）

`BernoulliNB` - 伯努利朴素贝叶斯

```python
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB(
    alpha=1.0,
    binarize=0.0,           # 二值化阈值
    fit_prior=True
)
```

**适用场景**：二值特征（如文本存在与否）

### 3. 树模型补充

`DecisionTreeClassifier` - 决策树分类器

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion='gini',       # 分裂标准：'gini'或'entropy'
    splitter='best',        # 分裂策略：'best'或'random'
    max_depth=None,         # 树最大深度
    min_samples_split=2,    # 内部节点最小样本数
    min_samples_leaf=1,     # 叶节点最小样本数
    min_weight_fraction_leaf=0.0,
    max_features=None,      # 寻找最优分裂时考虑的特征数
    random_state=42,
    ccp_alpha=0.0           # 最小成本复杂度剪枝
)
```

`ExtraTreesClassifier` - 极端随机树分类器

```python
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(
    n_estimators=100,       # 树的数量
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='auto',    # 分裂时的特征数：'auto','sqrt','log2'
    bootstrap=False,        # 是否自助采样
    n_jobs=-1,
    random_state=42
)
```

**特点**：比随机森林更随机，速度更快，可能精度略低

`AdaBoostClassifier` - AdaBoost分类器

```python
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(
    base_estimator=None,    # 基础学习器，默认决策树
    n_estimators=50,        # 基学习器数量
    learning_rate=1.0,      # 学习率
    algorithm='SAMME.R',    # 'SAMME'或'SAMME.R'
    random_state=42
)
```

`GradientBoostingClassifier` - 梯度提升分类器

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    loss='deviance',        # 损失函数：'deviance','exponential'
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,          # 样本采样比例
    criterion='friedman_mse',# 分裂标准
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=3,
    max_features=None,
    random_state=42
)
```

### 4. 判别分析

`LinearDiscriminantAnalysis` - 线性判别分析

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis(
    solver='svd',           # 求解器：'svd','lsqr','eigen'
    shrinkage=None,         # 收缩参数：'auto'或0-1
    priors=None,            # 先验概率
    n_components=None,      # 降维后的维度
    store_covariance=False, # 是否存储协方差矩阵
    tol=1e-4
)
```

`QuadraticDiscriminantAnalysis` - 二次判别分析

```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

model = QuadraticDiscriminantAnalysis(
    priors=None,
    reg_param=0.0,          # 正则化参数
    store_covariance=False,
    tol=1e-4
)
```

**区别**：LDA假设各类协方差相同，QDA假设不同

## 回归（Regression）补充 

### 1. 线性回归补充

`Lasso` - L1正则化线性回归

```python
from sklearn.linear_model import Lasso

model = Lasso(
    alpha=1.0,              # 正则化强度
    fit_intercept=True,
    max_iter=1000,
    tol=1e-4,
    warm_start=False,       # 是否热启动
    selection='cyclic',     # 'cyclic'或'random'
    random_state=42
)
```

**特点**：产生稀疏解，适合特征选择

`ElasticNet` - L1+L2正则化

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(
    alpha=1.0,
    l1_ratio=0.5,           # L1比例：0(L2)到1(L1)
    fit_intercept=True,
    max_iter=1000,
    tol=1e-4,
    random_state=42
)
```

`SGDRegressor` - 随机梯度下降回归

```python
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(
    loss='squared_error',   # 损失函数
    penalty='l2',
    alpha=0.0001,
    l1_ratio=0.15,
    max_iter=1000,
    tol=1e-3,
    learning_rate='invscaling',
    eta0=0.01,
    power_t=0.25,
    random_state=42
)
```

`HuberRegressor` - 鲁棒回归

```python
from sklearn.linear_model import HuberRegressor

model = HuberRegressor(
    epsilon=1.35,           # 定义离群点的阈值
    max_iter=100,
    alpha=0.0001,
    warm_start=False,
    tol=1e-5
)
```

**特点**：对异常值不敏感

`RANSACRegressor` - RANSAC回归

```python
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

model = RANSACRegressor(
    base_estimator=LinearRegression(),  # 基础估计器
    min_samples=None,       # 随机子集最小样本数
    residual_threshold=None,# 离群点阈值
    max_trials=100,         # 最大迭代次数
    stop_score=0.85,        # 停止分数阈值
    stop_probability=0.99,  # 停止概率
    random_state=42
)
```

**特点**：鲁棒回归，自动排除离群点

### 2. 树与集成回归

`DecisionTreeRegressor` - 决策树回归

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    criterion='squared_error', # 分裂标准
    splitter='best',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

`ExtraTreesRegressor` - 极端随机树回归

```python
from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor(
    n_estimators=100,
    criterion='squared_error',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='auto',
    bootstrap=False,
    n_jobs=-1,
    random_state=42
)
```

`GradientBoostingRegressor` - 梯度提升回归

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    loss='squared_error',   # 损失函数
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,
    criterion='friedman_mse',
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=3,
    random_state=42
)
```

`AdaBoostRegressor` - AdaBoost回归

```python
from sklearn.ensemble import AdaBoostRegressor

model = AdaBoostRegressor(
    base_estimator=None,
    n_estimators=50,
    learning_rate=1.0,
    loss='linear',          # 损失函数：'linear','square','exponential'
    random_state=42
)
```

### 3. 支持向量回归

`SVR` - 支持向量回归

```python
from sklearn.svm import SVR

model = SVR(
    kernel='rbf',           # 核函数
    C=1.0,                  # 正则化参数
    epsilon=0.1,            # epsilon不敏感损失
    gamma='scale',          # 核系数
    shrinking=True,         # 是否使用收缩启发式
    tol=1e-3,
    max_iter=-1             # 无限制
)
```

`LinearSVR` - 线性支持向量回归

```python
from sklearn.svm import LinearSVR

model = LinearSVR(
    epsilon=0.0,
    C=1.0,
    loss='epsilon_insensitive', # 损失函数
    fit_intercept=True,
    intercept_scaling=1.0,
    dual=True,
    tol=1e-4,
    max_iter=1000,
    random_state=42
)
```

## 聚类（Clustering）补充 

### 1. 基于划分的聚类

`MiniBatchKMeans` - 小批量K均值

```python
from sklearn.cluster import MiniBatchKMeans

model = MiniBatchKMeans(
    n_clusters=8,           # 聚类数
    init='k-means++',       # 初始化方法
    max_iter=100,
    batch_size=100,         # 小批量大小
    tol=0.0,
    max_no_improvement=10,  # 无改进最大迭代次数
    init_size=None,         # 初始化采样数
    n_init=3,
    random_state=42
)
```

**优点**：适合大规模数据，内存效率高

### 2. 层次聚类

`AgglomerativeClustering` - 凝聚层次聚类

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=2,           # 聚类数
    metric='euclidean',     # 距离度量
    memory=None,            # 缓存目录
    connectivity=None,      # 连通性约束
    compute_full_tree='auto',# 是否计算完整树
    linkage='ward',         # 链接准则：'ward','complete','average','single'
    distance_threshold=None # 距离阈值
)
```

### 3. 模型聚类

`GaussianMixture` - 高斯混合模型

```python
from sklearn.mixture import GaussianMixture

model = GaussianMixture(
    n_components=1,         # 高斯成分数
    covariance_type='full', # 协方差类型：'full','tied','diag','spherical'
    tol=1e-3,
    reg_covar=1e-6,         # 协方差正则化
    max_iter=100,
    n_init=1,
    init_params='kmeans',   # 初始化方法
    random_state=42
)
```

`BayesianGaussianMixture` - 贝叶斯高斯混合

```python
from sklearn.mixture import BayesianGaussianMixture

model = BayesianGaussianMixture(
    n_components=1,
    covariance_type='full',
    tol=1e-3,
    reg_covar=1e-6,
    max_iter=100,
    n_init=1,
    init_params='kmeans',
    weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=None,
    mean_precision_prior=None,
    random_state=42
)
```

**特点**：自动确定聚类数

### 4. 图与谱方法

`SpectralClustering` - 谱聚类

```python
from sklearn.cluster import SpectralClustering

model = SpectralClustering(
    n_clusters=8,
    eigen_solver=None,      # 特征求解器
    n_components=None,      # 特征向量数
    random_state=42,
    n_init=10,
    gamma=1.0,              # RBF核参数
    affinity='rbf',         # 相似度度量
    n_neighbors=10,
    assign_labels='kmeans'  # 分配标签方法
)
```

`AffinityPropagation` - 亲和传播

```python
from sklearn.cluster import AffinityPropagation

model = AffinityPropagation(
    damping=0.5,            # 阻尼系数(0.5-1)
    max_iter=200,
    convergence_iter=15,    # 收敛迭代次数
    preference=None,        # 偏好参数
    affinity='euclidean',   # 相似度度量
    random_state=42
)
```

**特点**：自动确定聚类数，适合中小规模数据

`MeanShift` - 均值漂移

```python
from sklearn.cluster import MeanShift

model = MeanShift(
    bandwidth=None,         # 带宽
    seeds=None,             # 初始种子
    bin_seeding=False,      # 是否使用分箱加速
    min_bin_freq=1,         # 最小分箱频率
    cluster_all=True,       # 是否聚类所有点
    n_jobs=-1
)
```

## 降维（Dimensionality Reduction）补充 

### 1. 线性降维

`TruncatedSVD` - 截断SVD

```python
from sklearn.decomposition import TruncatedSVD

model = TruncatedSVD(
    n_components=2,         # 降维后的维度
    algorithm='randomized', # 算法：'randomized','arpack'
    n_iter=5,
    random_state=42,
    tol=0.0
)
```

**适用场景**：稀疏矩阵，文本数据

`FactorAnalysis` - 因子分析

```python
from sklearn.decomposition import FactorAnalysis

model = FactorAnalysis(
    n_components=2,         # 因子数
    tol=1e-2,
    max_iter=1000,
    noise_variance_init=None,
    svd_method='randomized',
    random_state=42
)
```

`FastICA` - 快速独立成分分析

```python
from sklearn.decomposition import FastICA

model = FastICA(
    n_components=None,      # 成分数
    algorithm='parallel',   # 算法：'parallel','deflation'
    whiten=True,            # 是否白化
    fun='logcosh',          # 非线性函数
    max_iter=200,
    tol=1e-4,
    random_state=42
)
```

**特点**：盲源分离，寻找独立成分

### 2. 非线性降维

`TSNE` - t分布随机邻域嵌入

```python
from sklearn.manifold import TSNE

model = TSNE(
    n_components=2,         # 降维维度
    perplexity=30.0,        # 困惑度(通常5-50)
    early_exaggeration=12.0,# 早期放大
    learning_rate=200.0,    # 学习率
    n_iter=1000,            # 迭代次数
    min_grad_norm=1e-7,     # 最小梯度范数
    metric='euclidean',     # 距离度量
    init='random',          # 初始化
    random_state=42
)
```

**适用场景**：高维数据可视化，计算成本高

`Isomap` - 等距映射

```python
from sklearn.manifold import Isomap

model = Isomap(
    n_components=2,
    n_neighbors=5,          # 邻域数
    metric='minkowski',     # 距离度量
    p=2,                    # 闵可夫斯基距离参数
    neighbors_algorithm='auto'
)
```

`LocallyLinearEmbedding` - 局部线性嵌入

```python
from sklearn.manifold import LocallyLinearEmbedding

model = LocallyLinearEmbedding(
    n_components=2,
    n_neighbors=5,
    reg=1e-3,               # 正则化参数
    eigen_solver='auto',
    tol=1e-6,
    max_iter=100,
    method='standard',      # 方法：'standard','modified','hessian','ltsa'
    random_state=42
)
```

`MDS` - 多维尺度分析

```python
from sklearn.manifold import MDS

model = MDS(
    n_components=2,
    metric=True,            # 是否度量MDS
    n_init=4,
    max_iter=300,
    eps=1e-3,
    random_state=42,
    n_jobs=-1
)
```

## 模型选择（Model Selection）补充 

### 1. 超参数搜索

`RandomizedSearchCV` - 随机搜索交叉验证

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'C': uniform(0.1, 10),
    'gamma': uniform(0.01, 1),
    'kernel': ['rbf', 'linear']
}

search = RandomizedSearchCV(
    estimator=SVC(),
    param_distributions=param_dist,
    n_iter=100,             # 随机采样次数
    scoring=None,           # 评分指标
    n_jobs=-1,
    cv=5,                   # 交叉验证折数
    verbose=0,
    random_state=42,
    return_train_score=False
)
```

`HalvingGridSearchCV` - 减半网格搜索

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

search = HalvingGridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={'max_depth': [3, 5, 10], 'n_estimators': [10, 50, 100]},
    factor=3,               # 每轮资源增长因子
    resource='n_samples',   # 资源类型：'n_samples','n_iterations'
    max_resources='auto',   # 最大资源
    min_resources='exhaust',# 最小资源
    aggressive_elimination=False,
    cv=5,
    scoring=None,
    n_jobs=-1,
    random_state=42
)
```

`HalvingRandomSearchCV` - 减半随机搜索

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

search = HalvingRandomSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_dist,
    n_candidates='exhaust', # 初始候选数
    factor=3,
    resource='n_samples',
    max_resources='auto',
    min_resources='smallest',
    aggressive_elimination=False,
    cv=5,
    scoring=None,
    n_jobs=-1,
    random_state=42
)
```

### 2. 交叉验证

`cross_val_score` - 交叉验证评分

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    estimator=model,
    X=X,
    y=y,
    scoring=None,           # 评分指标
    cv=5,                   # 交叉验证策略
    n_jobs=-1,
    verbose=0,
    fit_params=None,
    pre_dispatch='2*n_jobs',
    error_score='raise'
)
```

`cross_validate` - 交叉验证（多指标）

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    estimator=model,
    X=X,
    y=y,
    scoring=['accuracy', 'precision', 'recall'],
    cv=5,
    return_train_score=False,
    return_estimator=False,
    n_jobs=-1
)
```

`KFold` - K折交叉验证

```python
from sklearn.model_selection import KFold

kf = KFold(
    n_splits=5,             # 折数
    shuffle=False,          # 是否洗牌
    random_state=None
)

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
```

`StratifiedKFold` - 分层K折交叉验证

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=5,
    shuffle=False,
    random_state=None
)
```

**特点**：保持每折的类别比例

`TimeSeriesSplit` - 时间序列分割

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(
    n_splits=5,
    max_train_size=None,    # 最大训练集大小
    test_size=None,
    gap=0                   # 训练测试间隔
)
```

## 数据预处理（Preprocessing）补充 

### 1. 数值变换

`RobustScaler` - 鲁棒缩放

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler(
    with_centering=True,    # 是否中心化
    with_scaling=True,      # 是否缩放
    quantile_range=(25.0, 75.0)  # 分位数范围
)
```

**特点**：对异常值不敏感

`MaxAbsScaler` - 最大绝对值缩放

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
```

**特点**：将数据缩放到[-1, 1]，不破坏稀疏性

`Normalizer` - 样本归一化

```python
from sklearn.preprocessing import Normalizer

scaler = Normalizer(
    norm='l2'               # 范数类型：'l1','l2','max'
)
```

`PowerTransformer` - 幂变换

```python
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer(
    method='yeo-johnson',   # 方法：'yeo-johnson','box-cox'
    standardize=True,       # 是否标准化
    copy=True
)
```

**特点**：使数据更接近正态分布

`QuantileTransformer` - 分位数变换

```python
from sklearn.preprocessing import QuantileTransformer

transformer = QuantileTransformer(
    n_quantiles=1000,       # 分位数数
    output_distribution='uniform', # 输出分布：'uniform','normal'
    ignore_implicit_zeros=False,
    subsample=10000,        # 子样本数
    random_state=42
)
```

### 2. 编码

`LabelEncoder` - 标签编码

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

**注意**：仅用于目标变量，特征编码用OrdinalEncoder

`OrdinalEncoder` - 序数编码

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories='auto',      # 类别列表
    dtype=np.float64
)
X_encoded = encoder.fit_transform(X)
```

`MultiLabelBinarizer` - 多标签二值化

```python
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(
    classes=None,
    sparse_output=False
)
X_bin = mlb.fit_transform(X)
```

### 3. 特征生成

`PolynomialFeatures` - 多项式特征

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2,               # 多项式次数
    interaction_only=False, # 是否只包含交互项
    include_bias=True,      # 是否包含偏置项
    order='C'               # 输出顺序
)
X_poly = poly.fit_transform(X)
```

## 特征工程（Feature Engineering）

### 特征选择

`SelectKBest` - 选择K个最佳特征

```python
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(
    score_func=chi2,        # 评分函数
    k=10                    # 选择特征数
)
X_new = selector.fit_transform(X, y)
```

`chi2` - 卡方检验

```python
from sklearn.feature_selection import chi2

chi2_scores, p_values = chi2(X, y)
```

`f_classif` - 方差分析的F值

```python
from sklearn.feature_selection import f_classif

f_scores, p_values = f_classif(X, y)
```

`RFE` - 递归特征消除

```python
from sklearn.feature_selection import RFE

selector = RFE(
    estimator=LogisticRegression(),  # 基础估计器
    n_features_to_select=5,          # 选择特征数
    step=1,                          # 每步移除特征数
    verbose=0
)
X_new = selector.fit_transform(X, y)
```

`SelectFromModel` - 基于模型的特征选择

```python
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(
    estimator=RandomForestClassifier(),  # 基础估计器
    threshold=None,               # 阈值：'mean','median',float
    prefit=False,                 # 是否已拟合
    norm_order=1,
    max_features=None
)
X_new = selector.fit_transform(X, y)
```

## 管道与组合（Pipeline & Composition）

`Pipeline` - 管道

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('clf', RandomForestClassifier())
])

# 网格搜索
param_grid = {
    'pca__n_components': [2, 3, 5],
    'clf__n_estimators': [50, 100, 200]
}
```

`FeatureUnion` - 特征联合

```python
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

union = FeatureUnion([
    ("pca", PCA(n_components=2)),
    ("select_best", SelectKBest(k=1))
])

pipe = Pipeline([
    ('features', union),
    ('clf', RandomForestClassifier())
])
```

`ColumnTransformer` - 列转换器

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'income']),
        ('cat', OneHotEncoder(), ['gender', 'city'])
    ],
    remainder='drop',        # 剩余列处理：'drop','passthrough'
    sparse_threshold=0.3
)
```

`make_column_selector` - 创建列选择器

```python
from sklearn.compose import make_column_selector

num_selector = make_column_selector(dtype_include=np.number)
cat_selector = make_column_selector(dtype_include=object)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_selector),
    ('cat', OneHotEncoder(), cat_selector)
])
```

## 模型评估（Metrics）

### 1. 分类评估

#### 基础指标

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# 准确率
acc = accuracy_score(y_true, y_pred)

# 精确率（宏观/微观/加权平均）
precision = precision_score(y_true, y_pred, average='weighted')

# 召回率
recall = recall_score(y_true, y_pred, average='weighted')

# F1分数
f1 = f1_score(y_true, y_pred, average='weighted')

# ROC-AUC（二分类/多分类）
roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr')

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 分类报告
report = classification_report(y_true, y_pred)
```

### 2. 回归评估

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score
)

# 均方误差
mse = mean_squared_error(y_true, y_pred)

# 均方根误差
rmse = np.sqrt(mse)

# 平均绝对误差
mae = mean_absolute_error(y_true, y_pred)

# 平均绝对百分比误差
mape = mean_absolute_percentage_error(y_true, y_pred)

# R²分数
r2 = r2_score(y_true, y_pred)
```

## 异常检测（Outlier Detection）

`IsolationForest` - 孤立森林

```python
from sklearn.ensemble import IsolationForest

detector = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination='auto',    # 异常值比例
    max_features=1.0,
    bootstrap=False,
    n_jobs=-1,
    random_state=42,
    verbose=0
)
```

`LocalOutlierFactor` - 局部离群因子

```python
from sklearn.neighbors import LocalOutlierFactor

detector = LocalOutlierFactor(
    n_neighbors=20,
    algorithm='auto',
    leaf_size=30,
    metric='minkowski',
    p=2,
    contamination='auto',
    novelty=False           # 是否用于新样本检测
)
```

`OneClassSVM` - 一类支持向量机

```python
from sklearn.svm import OneClassSVM

detector = OneClassSVM(
    kernel='rbf',
    degree=3,
    gamma='scale',
    coef0=0.0,
    tol=1e-3,
    nu=0.5,                 # 异常值上限比例
    shrinking=True,
    cache_size=200,
    max_iter=-1
)
```

`EllipticEnvelope` - 椭圆包络

```python
from sklearn.covariance import EllipticEnvelope

detector = EllipticEnvelope(
    store_precision=True,
    assume_centered=False,
    support_fraction=None,   # 支持点比例
    contamination=0.1,
    random_state=42
)
```

## 半监督/弱监督（进阶）

 `LabelPropagation` - 标签传播
 
```python
from sklearn.semi_supervised import LabelPropagation

model = LabelPropagation(
    kernel='rbf',           # 核函数：'rbf','knn'
    gamma=20,               # RBF核参数
    n_neighbors=7,          # KNN近邻数
    alpha=0.2,              # 钳位因子
    max_iter=1000,
    tol=1e-3,
    n_jobs=-1
)
```

`LabelSpreading` - 标签传播（正则化）

```python
from sklearn.semi_supervised import LabelSpreading

model = LabelSpreading(
    kernel='rbf',
    gamma=20,
    n_neighbors=7,
    alpha=0.2,              # 正则化参数
    max_iter=1000,
    tol=1e-3,
    n_jobs=-1
)
```