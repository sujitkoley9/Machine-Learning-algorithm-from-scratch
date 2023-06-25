#-----------------------------------------------------------------------------------------------#
#------------------------k nearest neighbors : Written by sujit --------------------------------#
#-----------------------------------------------------------------------------------------------#

# Importing packages
import math
import numpy as np
from sklearn.model_selection import train_test_split


class KNN():
    def __init__(self,k):
        self.k = k

    # calculate euclidean distance
    def euclidean_distance(self,x1, x2):
        """ Calculates the l2 distance between two vectors """
        distance = 0
        # Squared distance between each coordinate
        for i in range(len(x1)):
            distance += pow((x1[i] - x2[i]), 2)
        return math.sqrt(distance)

    # vote
    def vote(self, neighbor_labels):
        """ Return the most common class among the neighbor samples """
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()

    # fit
    def fit(self, X_test, X_train, y_train):
        self.y_pred_neighbors = np.empty((X_test.shape[0],self.k))
        # Determine the class of each sample
        for i, test_sample in enumerate(X_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            idx = np.argsort([self.euclidean_distance(test_sample, x) for x in X_train])[:self.k]
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors = np.array([y_train[i] for i in idx])

            # Label sample as the most common class label
            self.y_pred_neighbors[i,:] = k_nearest_neighbors

    # Predict
    def predict(self):
        y_preds=[]
        for i in self.y_pred_neighbors:
            y_preds.append(self.vote(i))

        return np.array(y_preds)





# Run random forest
def KNN_classifier():
    # Load record set
    record = np.loadtxt("spam-data.txt", delimiter=" ")
    print('Number of records: {}'.format(len(record)))

    X = record[:, :-1]
    y = record[:, -1]

    # Split training/test sets
    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42)


    # # run naive bayes classifier
    knn = KNN(k=5)

    knn.fit(X_test, X_train, y_train)

    # predictions
    predictions = knn.predict()

    # accuracy
    accuracy = (predictions == y_test).sum()/predictions.shape[0]

    # print accuracy
    print('accuracy:', accuracy)


# main method
if __name__ == "__main__":
    KNN_classifier()

