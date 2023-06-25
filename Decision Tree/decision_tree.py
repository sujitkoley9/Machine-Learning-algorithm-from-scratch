
#-----------------------------------------------------------------------------------------------#
#------------------------Decision tree : Written by sujit --------------------------------------#
#-----------------------------------------------------------------------------------------------#

# Importing packages
import numpy as np
from numpy import log2
import pprint

# DecisionTree class
class DecisionTree():
     # Constructor
    def __init__(self, header, min_samples, max_depth, criteria, feature_index=''):
        self.header = header
        self.feature_index = feature_index
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.criteria = criteria
        self.counter = 0
        self.tree = {}

    # Leveling records as per count
    def get_class_record(self, record):

        target_varible = record[:, -1]
        unique_classes, unique_class_count = np.unique(
            target_varible, return_counts=True)

        index = unique_class_count.argmax()
        required_class = unique_classes[index]

        return required_class

    # Get potential split for each feature
    def get_potential_splits_each_feature(self, record):
        potential_splits_dic = {}
        no_of_feature = record.shape[1] - 1
        for col_index in range(no_of_feature):
            potential_splits_dic[col_index] = []
            values = record[:, col_index]
            unique_values = np.unique(values)
            for index in range(len(unique_values)):
                if index > 0:
                    cur_value = unique_values[index]
                    prev_value = unique_values[index - 1]
                    potential_split = (cur_value + prev_value) / 2

                    potential_splits_dic[col_index].append(potential_split)

        return potential_splits_dic

    # get split record based on boundary
    def split_record(self, record, split_col_index, split_value):
        record_below = record[record[:, split_col_index] <= split_value]
        record_above = record[record[:, split_col_index] > split_value]

        return record_below, record_above

    # Find entropy score
    def find_entropy(self, record_below, record_above):
        total = len(record_below) + len(record_above)
        prob_record_below = len(record_below) / total
        prob_record_above = len(record_above) / total

        # for record below
        target_variable = record_below[:, -1]
        _, counts = np.unique(target_variable, return_counts=True)
        probabilities = counts / counts.sum()
        entropy_below = sum(probabilities*-log2(probabilities))

        # for record above
        target_variable = record_above[:, -1]
        _, counts = np.unique(target_variable, return_counts=True)
        probabilities = counts / counts.sum()
        entropy_above = sum(probabilities*-log2(probabilities))

        entropy_score = prob_record_below*entropy_below + prob_record_above*entropy_above

        return entropy_score

    # find gini score
    def find_gini_score(self, record_below, record_above):
        n = len(record_below) + len(record_above)
        prob_record_below = len(record_below) / n
        prob_record_above = len(record_above) / n

        # for record below
        target_variable = record_below[:, -1]
        _, counts = np.unique(target_variable, return_counts=True)
        probabilities = counts / counts.sum()
        gini_below = sum(probabilities*probabilities)

        # for record above
        target_variable = record_above[:, -1]
        _, counts = np.unique(target_variable, return_counts=True)
        probabilities = counts / counts.sum()
        gini_above = sum(probabilities*probabilities)

        gini_score = prob_record_below*gini_below + prob_record_above*gini_above

        return gini_score


    # determin best split using entropy
    def determine_best_split_using_entropy(self, record, potential_splits):

        overall_entropy_score = float("inf")
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                record_below, record_above = self.split_record(
                    record, column_index, value)
                current_overall_entropy_score = self.find_entropy(
                    record_below, record_above)

                if current_overall_entropy_score <= overall_entropy_score:
                    overall_entropy_score = current_overall_entropy_score
                    best_split_column_index = column_index
                    best_split_value = value

        return best_split_column_index, best_split_value


    # determine best split using gini
    def determine_best_split_using_gini(self, record, potential_splits):

        overall_gini_score = float("-inf")

        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                record_below, record_above = self.split_record(
                    record, column_index, value)
                current_overall_gini_score = self.find_gini_score(
                    record_below, record_above)

                if current_overall_gini_score >= overall_gini_score:
                    overall_gini_score = current_overall_gini_score
                    best_split_column_index = column_index
                    best_split_value = value

        return best_split_column_index, best_split_value

    # Check purity of a node
    def check_pirity_of_node(self, training_set):
        unique_classes, _ = np.unique(
            training_set[:, -1], return_counts=True)

        if len(unique_classes) == 1:
            purity = True
        else:
            purity = False

        return purity

    # in case of no potential split found
    def check_potential_split(self, potential_splits):
        is_potential_split_found= False

        for _, value in potential_splits.items():
            if len(value)>0:
                is_potential_split_found =True
                break

        return is_potential_split_found



    # Training  decision tree
    def learn(self, training_set):
        potential_splits = self.get_potential_splits_each_feature(
             training_set)
        if not self.check_potential_split(potential_splits) or self.check_pirity_of_node(training_set) or (len(training_set) < self.min_samples) or self.counter >= self.max_depth:
            return self.get_class_record(training_set)  # get class of record

        self.counter += 1  # increment the counter to check max_depth is reached or not



        if self.criteria == 'entropy':
            split_column_index, split_value = self.determine_best_split_using_entropy(
                training_set, potential_splits)
        elif self.criteria == 'gini':
             split_column_index, split_value = self.determine_best_split_using_gini(
                 training_set, potential_splits)

        record_below, record_above = self.split_record(
            training_set, split_column_index, split_value)

        feature_name = self.header[split_column_index]
        # Condition
        condition = "{} <= {}".format(feature_name, split_value)
        tree = {}
        tree[condition] = []

        # add record below class
        tree[condition].append(self.learn(
            record_below))
        # add record above class
        tree[condition].append(self.learn(
            record_above))

        return tree


    # Fit the decision tree using X,y
    def fit(self,X,y):
        training_set = np.hstack((X, y))
        self.tree = self.learn(training_set)

    # identify class of test record set
    def classify(self, test_instance, tree):
        condition = list(tree.keys())[0]

        feature_name,spiliting_condition_value = condition.split("<=")
        feature_index = np.where(self.header == feature_name.strip())[0][0]
        test_val = test_instance[feature_index]

        # key form
        if test_val <= float(spiliting_condition_value):
            tree = list(tree.values())[0][0]
        else:
            tree = list(tree.values())[0][1]

        if type(tree) is dict:
            return self.classify(test_instance, tree)

        else:
            return tree

    # predict on test dataset
    def predict(self,test_set):
        results = []
        for instance in test_set:
            result = self.classify(instance, self.tree)
            results.append(result)

        return np.array(results)




