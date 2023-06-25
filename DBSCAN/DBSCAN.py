import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

class DBSCAN():
    def __init__(self, eps, minPts):
        self.eps = eps
        self.minPts = minPts
        self.already_visited=[]


    """ calculate euclidean distance """
    def euclidian_distance(self,a,b):
        dis = norm(a-b)
        return dis

    """ find neighbous of a points """
    def find_neighbours(self,sample_i):
        neighbours =[]
        for rec_i in range(len(self.X)):
            if rec_i != sample_i:
                if self.euclidian_distance(self.X[rec_i], self.X[sample_i]) <= self.eps:
                    neighbours.append(rec_i)

        return neighbours

    """ predict """
    def predict(self,X):
        self.X =X
        c=-1
        level = np.full(len(self.X),-1)

        for rec_i  in range(len(self.X)):

            if rec_i in self.already_visited:
                continue
            neighbours = self.find_neighbours(rec_i)
            if len(neighbours) >=self.minPts:
                c = c+1
                level[rec_i] = c
                self.already_visited.append(rec_i)
                s = neighbours
                for  rec_j  in s:
                    if rec_j in self.already_visited:
                        continue
                    level[rec_j] = c
                    self.already_visited.append(rec_j)
                    neighbours = self.find_neighbours(rec_j)
                    if len(neighbours) >= self.minPts:
                        s.extend(neighbours)

        return level



# main method
if __name__ == "__main__":
    records = pd.read_csv('Mall_Customers.csv')
    records_for_clustering = records.iloc[:, 3:].values

    dbscan_obj = DBSCAN(eps=10, minPts=2)
    level = dbscan_obj.predict(records_for_clustering)
    print(level)


    colors = ['r', 'g', 'b', 'y','c']
    fig, ax = plt.subplots()
    for i in [-1,0,1,2,3]:
            points = np.array([records_for_clustering[j] for j in range(
                len(records_for_clustering)) if level[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])



    fig.savefig('clustering.png', dpi=fig.dpi)








