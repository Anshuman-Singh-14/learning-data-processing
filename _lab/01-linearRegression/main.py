import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris_df = pd.read_csv('iris.csv')
iris_df.head()

data = load_iris()


#data.feature_names
#data.target_names
#data.target

x = data.data
y = data.target

y = y.reshape(-1, 1)


def plotPetalSepal(X):

    plt.figure(figsize=(18,8),dpi=100)
    plt.scatter(X.T[0],X.T[2])
    plt.title('IRIS Petal and sepal length', fontsize=20)
    plt.ylabel('Petal Length') 
    plt.xlabel('sepal length')

#plotPetalSepal(x)





