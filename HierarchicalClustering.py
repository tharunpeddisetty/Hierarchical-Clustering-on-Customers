import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 4 - Clustering/Section 24 - K-Means Clustering/Python/Mall_Customers.csv')

X = dataset.iloc[:,3:].values

#Using dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X, method='ward')) #method ward minimizes the variance inside clusters
plt.title('Dendrogram')
plt.xlabel('Customers') #observation points
plt.ylabel('Eucledian Distance')
plt.show()

#Training the Hierarchical clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward') #affinity is the type of disntance we consider for checking the variance
y_hc=hc.fit_predict(X)

#Visualizing the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100, c ='red',label='Cluster 1')#here x is annual income an Y is spending score. In X we select only the ones that have specific cluster index
#s is for size, c is for color
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100, c ='blue',label='Cluster 2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100, c ='green',label='Cluster 3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100, c ='cyan',label='Cluster 4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100, c ='magenta',label='Cluster 5') 
plt.title('Clusters of Customers')
plt.xlabel('Annual income in ($k)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()