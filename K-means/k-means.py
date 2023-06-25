import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm

class Kmeans():
    def __init__(self,k,max_iterations):
        self.k=k
        self.max_iterations = max_iterations

    """ calculate euclidean distance """
    def euclidian_distance(self,a,b):
        dis = norm(a-b)
        return dis

    """ initial random centroids """
    def initial_random_centroids(self,X):
        initial_centroids_index = np.random.choice(
            records_for_clustering.shape[0], number_of_clusters)

        initial_centroids = records_for_clustering[initial_centroids_index, :]
        return initial_centroids

    """ Predict """
    def predict(self,X):
        prev_centroids = self.initial_random_centroids(X)
        distance_matrix = np.empty(
            (X.shape[0], number_of_clusters))

        clustering_id = np.empty(records_for_clustering.shape[0])

        for _ in range(self.max_iterations):
            """ calculating clustering id """
            for i, record in enumerate(X):
                for j, centroid in enumerate(prev_centroids):
                    distance_matrix[i, j] = self.euclidian_distance(record, centroid)

                clustering_id[i] = np.argmin(distance_matrix[i])



            """  calculate new centroids """
            new_centroids = np.empty((self.k, X.shape[1]))
            for k in range(self.k):
                new_centroids[k] = np.mean(
                    X[np.where(clustering_id == k)], axis=0)

            diff = new_centroids - prev_centroids

            if not np.any(diff):
                break
            prev_centroids = new_centroids

        return clustering_id, new_centroids


# main method
if __name__ == "__main__":
    records = pd.read_csv('Mall_Customers.csv')
    records_for_clustering = records.iloc[:, 3:].values
    number_of_clusters = 5
    max_iter =300

    kmean_obj = Kmeans(number_of_clusters, max_iter)
    clustering_id, centroids = kmean_obj.predict(records_for_clustering)


    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    for i in range(number_of_clusters):
            points = np.array([records_for_clustering[j] for j in range(
                len(records_for_clustering)) if clustering_id[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])

    ax.scatter(centroids[:, 0],
               centroids[:, 1], marker='*', s=200, c='#050505')

    fig.savefig('clustering.png', dpi=fig.dpi)








