from sklearn import datasets, svm, linear_model

digits = datasets.load_digits()
X, y = digits.data, digits.target
split = int(len(X) * 0.8)

svm_acc = svm.SVC().fit(X[:split], y[:split]).score(X[split:], y[split:])
log_acc = linear_model.LogisticRegression(max_iter=10000).fit(X[:split], y[:split]).score(X[split:], y[split:])

print(f"SVM Accuracy: {svm_acc:.4f}\nLogistic Regression Accuracy: {log_acc:.4f}")