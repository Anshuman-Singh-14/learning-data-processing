from sklearn.linear_model import Perceptron
import numpy as np

# OR Gate Logic
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 1])

clf = Perceptron().fit(X, y)
print(f"Perceptron Predictions: {clf.predict(X)}")