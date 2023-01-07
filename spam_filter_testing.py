from statistics import mean, stdev
import numpy as np
import os
from sklearn import preprocessing
from spam_filter_classes import EmailAttributeList, EmailInfo
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from itertools import compress, product
import time
import io
from pyexcel_ods3 import save_data
from collections import OrderedDict

def combinations(items):
    return [set(compress(items,mask)) for mask in product(*[[0,1]]*len(items)) ]
    



if __name__ == "__main__":
    
    data = "AttributeData.ods"
    data_handler = EmailAttributeList()
    data_handler.load(data)
    
    possible_attributes = list(data_handler.get_email_keys())

    possible_attributes.remove("spam")
    possible_attributes.remove("id")
    possible_attributes.remove("raw")

    possible_tests = combinations(possible_attributes)
    possible_tests.remove(set())
    print(len(possible_tests))
    breakpoint()

    best_attributes_tree = np.zeros(10,dtype=object)
    best_attributes_bayes = np.zeros(10,dtype=object)

    best_scores_tree = np.zeros(10)
    best_scores_bayes = np.zeros(10)

    start_time = time.time()
    index = 0


    for test in possible_tests:
        index += 1
        X, y = data_handler.to_feature_vector(test, amount_per_class=320)

        skf = StratifiedKFold(n_splits=10)
    
        tree_clf = tree.DecisionTreeClassifier()
        bayes_clf = GaussianNB()

        accuracy_scores_tree_skf = []
        accuracy_scores_bayes_skf = []

        for train_index, test_index in skf.split(X, y):
            x_train_fold, x_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            tree_clf.fit(x_train_fold, y_train_fold)
            bayes_clf.fit(x_train_fold, y_train_fold)

            accuracy_scores_tree_skf.append(tree_clf.score(x_test_fold, y_test_fold))
            accuracy_scores_bayes_skf.append(bayes_clf.score(x_test_fold, y_test_fold))

        if mean(accuracy_scores_tree_skf) > best_scores_tree[0]:
            best_scores_tree[0] = mean(accuracy_scores_tree_skf)
            best_attributes_tree[0] = str(test)
            accuracy_sort = np.argsort(best_scores_tree, kind = "quicksort")
            best_scores_tree = best_scores_tree[accuracy_sort]
            best_attributes_tree = best_attributes_tree[accuracy_sort]
     
        
        if mean(accuracy_scores_bayes_skf) > best_scores_bayes[0]:
            best_scores_bayes[0] = mean(accuracy_scores_bayes_skf)
            best_attributes_bayes[0] = str(test)
            accuracy_sort = np.argsort(best_scores_bayes, kind = "quicksort")
            best_scores_bayes = best_scores_bayes[accuracy_sort]
            best_attributes_bayes = best_attributes_bayes[accuracy_sort]

            
    

    print('time of execution:', time.time() - start_time)
    print()
    print("BAYES SCORES: ")
    for i in range(len(best_scores_bayes)):
        print(best_scores_bayes[i])
        print(best_attributes_bayes[i])
        print()
    
  
    print()
    print()
    print("TREE SCORES:")
    for i in range(len(best_scores_tree)):
        print(best_scores_tree[i])
        print(best_attributes_tree[i])
        print()
    print()

    col1 = [["Accuracy_tree"]]
    col1.append(best_scores_tree[::-1].tolist())
    col2 = [["Attributes_tree"]]
    col2.append(best_attributes_tree[::-1].tolist())
    
    col3 = [["Accuracy_bayes"]]
    col3.append(best_scores_bayes[::-1].tolist())
    col4 = [["Attributes_bayes"]]
    col4.append(best_attributes_bayes[::-1].tolist())

    columns1 = col1 + col2
    columns2 = col3 + col4
    save = OrderedDict()
    save.update({"Sheet 1": columns1})
    save.update({"Sheet 2": columns2})
    save_data("BestAttributesTREE_BAYES.ods", save)



    










