import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

iris = pd.read_csv("iris2d.csv")
print(iris.head())
f1 = iris['petallength'].values
f2 = iris['petalwidth'].values
X = iris.iloc[:, 0:2].values

plt.scatter(f1, f2, c='black', s=7)
plt.show()
print(X)

kmeans = KMeans(n_clusters=3, init='random')

kmeans.fit(X)
print(kmeans.cluster_centers_)

distance = kmeans.fit_transform(X)

labels = kmeans.labels_

print(labels)
#for x in labels:
#    print(x)

plt.scatter(f1,f2, c=kmeans.labels_, cmap='rainbow')
plt.show()