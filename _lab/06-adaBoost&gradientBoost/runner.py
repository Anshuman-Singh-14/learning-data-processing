from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=100, n_features=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"AdaBoost: {AdaBoostClassifier().fit(X_train, y_train).score(X_test, y_test):.2f}")
print(f"Gradient Boost: {GradientBoostingClassifier().fit(X_train, y_train).score(X_test, y_test):.2f}")