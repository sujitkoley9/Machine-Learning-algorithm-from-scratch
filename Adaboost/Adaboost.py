#-----------------------------------------------------------------------------------------------#
#------------------------Random  forest : Written by sujit -------------------------------------#
#-----------------------------------------------------------------------------------------------#

# Importing packages
import numpy as np
import math
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
import pprint
import pandas as pd

# AdaBoost class
class  AdaBoost:
    # Constructor
    def __init__(self, header, min_samples, max_depth, criteria, max_feature, no_of_treess):
        self.header = header
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.criteria = criteria
        self.max_feature = max_feature
        self.no_of_trees = no_of_treess
        self.addboost_trees =[]

    # get botstrap sample
    def get_boot_strap_sample(self,X,y,prob):
        row,_ = X.shape
        boot_strap_sample_index = np.random.choice(row, row, replace=True,p=prob)
        return X[boot_strap_sample_index], y[boot_strap_sample_index]

    # fit with X,y
    def fit(self, X, y):

        # Initialize weights to 1/N
        total_rows = X.shape[0]
        w = np.full(total_rows, (1 / total_rows))

        for _ in range(self.no_of_trees):
            X1, y1 = self.get_boot_strap_sample(X, y, w)  # botstarp sample
            tree = DecisionTree(
                header=self.header, min_samples=self.min_samples, max_depth=self.max_depth,
                criteria=self.criteria)

            y1 = y1.reshape(y1.shape[0], 1)
            tree.fit(X1, y1)
            error, predictions = self.calculate_error(w, tree, X, y)
            if error > .5:
                break
            # update weight
            w, alpha = self.update_weight(w,error, predictions,y )
            self.addboost_trees.append((tree, alpha))


    #-- calculate error
    def calculate_error(self, w, tree,X,y):
        predictions = tree.predict(X)
        error = w[(predictions != y)].sum()
        return error, predictions

    #-- update weight
    def update_weight(self, w, error, predictions, y):

        # Alpha is also an approximation of this classifier's proficiency
        alpha = 0.5 * math.log((1.0 - error) / (error + 1e-10))
        # Calculate new weights
        # Missclassified samples gets larger weights and correctly classified samples smaller
        w *= np.exp(-alpha * y * predictions)
        # Normalize to one
        w /= np.sum(w)

        return w, alpha

    # prediction on new data set
    def predict(self,test_set):
        predict_temp = np.full(test_set.shape[0],0.0)

        for tree,alpha in self.addboost_trees:
            predition_from_tree = tree.predict(test_set)
            predict_temp += alpha*predition_from_tree

        predictions = np.where(predict_temp>=0 ,1,-1)

        return predictions


# Run random forest
def run_adaboost():
    # Load record set
    record = np.loadtxt("spam-data.txt", delimiter=" ")
    print('Number of records: {}'.format(len(record)))

    X = record[:, :-1]
    y = record[:, -1]

    y= np.where(y==0,-1,1)

    # Split training/test sets
    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42)


    header = np.array(['column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7',
              'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14',
              'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21',
              'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28',
              'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35',
              'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42',
              'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49',
              'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55', 'column_56', 'column_57'
              ])

    no_of_toatal_feature = len(header)
    max_feature = int(np.sqrt(no_of_toatal_feature))
    min_samples =2
    max_depth=200
    criteria = 'entropy'
    no_of_treess=5

    # # run random forest
    adaboost_algo = AdaBoost(header, min_samples, max_depth,
                      criteria, max_feature, no_of_treess)

    adaboost_algo.fit(X_train, y_train)

    # predictions
    predictions = adaboost_algo.predict(X_test)

    # accuracyq
    accuracy = (predictions == y_test).sum()/predictions.shape[0]

    # print accuracy
    print('accuracy:', accuracy)




# main method
if __name__ == "__main__":
    run_adaboost()

