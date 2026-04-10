from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60)
kmeans = KMeans(n_clusters=4).fit(X)

plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='viridis')
plt.show()
print("Clustering complete. Centroids found at:\n", kmeans.cluster_centers_)