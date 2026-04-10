from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
preds = knn.predict(X_test)

for i in range(len(preds)):
    status = "Correct" if preds[i] == y_test[i] else "Wrong"
    print(f"Pred: {preds[i]}, Actual: {y_test[i]} - {status}")