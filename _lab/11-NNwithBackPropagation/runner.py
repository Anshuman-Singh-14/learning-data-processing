from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100)
# hidden_layer_sizes defines the ANN architecture
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000).fit(X, y)

print(f"ANN (Backprop) Score: {mlp.score(X, y):.2f}")