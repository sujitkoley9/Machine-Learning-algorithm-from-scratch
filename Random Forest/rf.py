#-----------------------------------------------------------------------------------------------#
#------------------------Random  forest : Written by sujit -------------------------------------#
#-----------------------------------------------------------------------------------------------#

# Importing packages
import numpy as np
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
import pprint

# Random Forest class
class RandomForest:
    # Constructor
    def __init__(self, header, min_samples, max_depth, criteria, max_feature, no_of_treess):
        self.header = header
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.criteria = criteria
        self.max_feature = max_feature
        self.no_of_trees = no_of_treess
        self.random_forest =[]

    # get botstrap sample
    def get_boot_strap_sample(self,X,y):
        row,_ = X.shape
        boot_strap_sample_index = np.random.choice(row, row, replace=True)
        return X[boot_strap_sample_index], y[boot_strap_sample_index]

    # get random features
    def get_random_feature(self, no_of_toatal_feature):
        selected_feature_index = np.random.choice(
            np.arange(no_of_toatal_feature), self.max_feature,replace=False)

        return selected_feature_index

    # fit with X,y
    def fit(self, X, y):
        no_of_toatal_feature = X.shape[1] # total feature
        for _ in range(self.no_of_trees):
            random_feature = self.get_random_feature(no_of_toatal_feature)

            header = self.header[random_feature] # select header based on randomness

            tree = DecisionTree(
                header=header, feature_index=random_feature, min_samples=self.min_samples, max_depth=self.max_depth,
                criteria=self.criteria)

            X1, y1 = self.get_boot_strap_sample(X, y) # botstarp sample
            X1 = X1[:, random_feature]
            tree.fit(X1, y1)
            self.random_forest.append(tree)

    # prediction on new data set
    def predict(self,test_set):
        y_preds = np.empty((test_set.shape[0], len(self.random_forest)))
        # each tree make a prediction on the data
        for i, tree in enumerate(self.random_forest):
            # Indices of the features that the tree has trained on
            idx = tree.feature_index
            # Make a prediction from tree
            prediction = tree.predict(test_set[:, idx])
            y_preds[:, i] = prediction

        y_pred = []
        # For each sample data
        for sample_predictions in y_preds:
            # Select the most common class prediction
            y_pred.append(np.bincount(
                sample_predictions.astype('int')).argmax())
        return np.array(y_pred)


# Run random forest
def run_random_forest():
    # Load record set
    record = np.loadtxt("spam-data.txt", delimiter=" ")
    print('Number of records: {}'.format(len(record)))

    X = record[:, :-1]
    y = record[:, -1].reshape(record.shape[0],1)


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
    no_of_treess=100

    # # run random forest
    rf = RandomForest(header, min_samples, max_depth,
                      criteria, max_feature, no_of_treess)

    rf.fit(X_train, y_train)

    # predictions
    predictions= rf.predict(X_test)

    # accuracy
    accuracy = (predictions == y_test.flatten()).sum()/predictions.shape[0]

    # print accuracy
    print('accuracy:', accuracy)


# main method
if __name__ == "__main__":
    run_random_forest()

