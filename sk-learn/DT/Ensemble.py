from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score

wine = load_wine()

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=0)
clf = RandomForestClassifier(n_estimators=50, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
print(dtc.score(X_test, y_test))

# 这是10折交叉验证
print(cross_val_score(clf, wine.data, wine.target, cv=10))
print(cross_val_score(dtc, wine.data, wine.target, cv=10))
# 这是直接使用
print(cross_val_score(RandomForestClassifier(n_estimators=50, random_state=0), wine.data, wine.target, cv=10))
print(cross_val_score(DecisionTreeClassifier(random_state=0), wine.data, wine.target, cv=10))
