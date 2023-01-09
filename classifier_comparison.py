from statistics import mean, stdev
import numpy as np
import os
from sklearn import preprocessing
from spam_filter_classes import EmailAttributeList, EmailInfo
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from itertools import compress, product
from sklearn.model_selection import train_test_split
import time
import io
from collections import OrderedDict
import matplotlib.pyplot as plt

from math import ceil
from math import sqrt
import scipy.stats as skp

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from spam_filter_training import get_cost_table, accuracy, precision, recall, f1, total_cost_ratio

if __name__ == "__main__":
    data = "AttributeData.ods"
    data_handler = EmailAttributeList()
    data_handler.load(data)
    data_handler.restrict_to_amount_per_class(500)

    best_smoothings = [0.000000007822, 0.000000009604, 0.000000009802]
    best_depths = [23, 17, 24]

    best_bayes_attributes = [
        ('SCONJ_relative', 'LIX', 'Unknown_tags_relative', 'INTJ_relative', 'PART_relative', 'DET_relative', 'PROPN_relative', 'ADJ_relative', 'polarity', 'capital_words_relative', 'ADP_relative', 'SYM_relative'),
        ('SCONJ_relative', 'LIX', 'Unknown_tags_relative', 'INTJ_relative', 'PART_relative', 'DET_relative', 'PROPN_relative', 'ADJ_relative', 'polarity', 'capital_words_relative', 'ADP_relative', 'SYM_relative'),
        ('SCONJ_relative', 'LIX', 'Unknown_tags_relative', 'INTJ_relative', 'PART_relative', 'DET_relative', 'PROPN_relative', 'ADJ_relative', 'polarity', 'capital_words_relative', 'ADP_relative', 'SYM_relative')
    ]

    best_tree_attributes = [
        ('SCONJ_relative', 'LIX', 'Unknown_tags_relative', 'PART_relative', 'NOUN_relative', 'VERB_relative', 'ADV_relative', 'AUX_relative', 'PRON_relative', 'DET_relative', 'polarity', 'capital_words_relative', 'ADP_relative', 'SYM_relative'),
        ('SCONJ_relative', 'LIX', 'Unknown_tags_relative', 'PART_relative', 'NOUN_relative', 'VERB_relative', 'ADV_relative', 'AUX_relative', 'PRON_relative', 'DET_relative', 'polarity', 'capital_words_relative', 'ADP_relative', 'SYM_relative'),
        ('SCONJ_relative', 'LIX', 'Unknown_tags_relative', 'PART_relative', 'NOUN_relative', 'VERB_relative', 'ADV_relative', 'AUX_relative', 'PRON_relative', 'DET_relative', 'polarity', 'capital_words_relative', 'ADP_relative', 'SYM_relative')
    ]

    bayes_clfs = []
    trees_clfs = []

    for i in range(len(best_depths)):
        trees_clfs.append(DecisionTreeClassifier(max_depth = best_depths[i]))
        bayes_clfs.append(GaussianNB(var_smoothing = best_smoothings[i]))
    
    classifier_tables_bayes = []
    classifier_tables_tree = []

    predictions_tree = []
    predictions_bayes = []
    correct_predictions = []

    data_length = len(data_handler.get_emails())

    training_length = int(data_length * 0.6)
    train = np.ones(training_length).tolist()
    test = np.zeros(data_length - training_length).tolist()
    test_train = np.array(test + train)
    np.random.shuffle(test_train)


    for i in range(len(best_bayes_attributes)):
        bayes_X, bayes_y = data_handler.to_feature_vector(best_bayes_attributes[i])
        tree_X, tree_y = data_handler.to_feature_vector(best_tree_attributes[i])
        X_train_bayes = []
        X_test_bayes = []
        y_train_bayes = []
        y_test_bayes = []

        gold_standard_test = []

        X_train_tree = []
        X_test_tree = []
        y_train_tree = []
        y_test_tree = []

        skf = StratifiedKFold(n_splits=20)
        tables_tree = []
        tables_bayes = []
        for train_index, test_index in skf.split(tree_X, tree_y):
            x_train_bayes_fold, x_test_bayes_fold = bayes_X[train_index], bayes_X[test_index]
            y_train_bayes_fold, y_test_bayes_fold = bayes_y[train_index], bayes_y[test_index]

            x_train_tree_fold, x_test_tree_fold = tree_X[train_index], tree_X[test_index]
            y_train_tree_fold, y_test_tree_fold = tree_y[train_index], tree_y[test_index]


            bayes_clfs[i].fit(x_train_bayes_fold, y_train_bayes_fold)
            trees_clfs[i].fit(x_train_tree_fold, y_train_tree_fold)

            bayes_y_pred = bayes_clfs[i].predict(x_test_bayes_fold)
            tree_y_pred = trees_clfs[i].predict(x_test_tree_fold)

            tables_tree.append(get_cost_table(tree_y_pred, y_test_tree_fold))
            tables_bayes.append(get_cost_table(bayes_y_pred, y_test_bayes_fold))

           # predictions_tree.append(tree_y_pred)
           # predictions_bayes.append(bayes_y_pred)
           # correct_predictions.append(gold_standard_test)
        classifier_tables_tree.append(tables_tree)
        classifier_tables_bayes.append(tables_bayes)

            
        """for j in range(len(test_train)):
            if test_train[j] == 1:
                X_train_bayes.append(bayes_X[j])
                y_train_bayes.append(bayes_y[j])

                X_train_tree.append(tree_X[j])
                y_train_tree.append(tree_y[j])
            else:
                X_test_bayes.append(bayes_X[j])
                y_test_bayes.append(bayes_y[j])

                X_test_tree.append(tree_X[j])
                y_test_tree.append(tree_y[j])
                gold_standard_test.append(tree_y[j])
        
        X_train_bayes = np.array(X_train_bayes)
        X_test_bayes = np.array(X_test_bayes)
        y_train_bayes = np.array(y_train_bayes)
        y_test_bayes = np.array(y_test_bayes)

        X_train_tree = np.array(X_train_tree)
        X_test_tree = np.array(X_test_tree)
        y_train_tree = np.array(y_train_tree)
        y_test_tree = np.array(y_test_tree)
        

        
        bayes_y_pred = bayes_clfs[i].predict(X_test_bayes)
        tree_y_pred = trees_clfs[i].predict(X_test_tree)

        predictions_tree.append(tree_y_pred)
        predictions_bayes.append(bayes_y_pred)
        correct_predictions.append(gold_standard_test)
        

        classifier_tables_bayes.append(get_cost_table(bayes_y_pred, y_test_bayes))
        classifier_tables_tree.append(get_cost_table(tree_y_pred, y_test_tree))
        """
    #saker kvar att göra:
        #få alla stats från klassifierare
    
    print("Tree scores: ")
    print()
    for i in range(len(classifier_tables_tree)):

        t_accuracy = mean([accuracy(table) for table in classifier_tables_tree[i]])
        w_accuracy = mean([accuracy(table, 99) for table in classifier_tables_tree[i]])
        f1_t = mean([f1(table) for table in classifier_tables_tree[i]])
        cost_ratio_w = mean([total_cost_ratio(table, 99) for table in classifier_tables_tree[i]])
        print(f"TREE {i} SCORES")
        print(f"Accuracy: {t_accuracy}")
        print(f"Weighted Accuracy: {w_accuracy}")
        print(f"F1: {f1_t}")
        print(f"Total cost ratio: {cost_ratio_w}")
        print()

    print()
    print()

    print("Bayes scores: ")
    print()
    for i in range(len(classifier_tables_bayes)):
        print(f"BAYES {i} SCORES")
        t_accuracy = mean([accuracy(table) for table in classifier_tables_bayes[i]])
        w_accuracy = mean([accuracy(table, 99) for table in classifier_tables_bayes[i]])
        f1_t = mean([f1(table) for table in classifier_tables_bayes[i]])
        cost_ratio_w = mean([total_cost_ratio(table, 99) for table in classifier_tables_bayes[i]])

        print(f"Accuracy: {t_accuracy}")
        print(f"Weighted Accuracy: {w_accuracy}")
        print(f"F1: {f1_t}")
        print(f"Total cost ratio: {cost_ratio_w}")
        print()

        

        


    """ for i in range(len(correct_predictions)):

        corr_both = 0
        bayes_corr_tree_wrong = 0
        tree_corr_bayes_wrong = 0
        wrong_both = 0
        for j in range(len(correct_predictions[i])):
            if predictions_tree[i][j] == predictions_bayes[i][j]:
                if predictions_bayes[i][j] == 1:
                    if correct_predictions[i][j] == 1:
                        corr_both += 1
                    else:
                        wrong_both += 1
                else:
                    if correct_predictions[i][j] == 0:
                        corr_both += 1
                    else:
                        wrong_both += 1
            else:
                if correct_predictions[i][j] == 1:
                    if predictions_tree[i][j] == 1:
                        tree_corr_bayes_wrong += 1
                    else:
                        bayes_corr_tree_wrong += 1
                else:
                    if predictions_tree[i][j] == 0:
                        tree_corr_bayes_wrong += 1
                    else:
                        bayes_corr_tree_wrong += 1

        table = np.array([[corr_both, bayes_corr_tree_wrong], [tree_corr_bayes_wrong, wrong_both]])
        print(table)

        N = bayes_corr_tree_wrong + tree_corr_bayes_wrong
        binom_dist = skp.binom(N, 0.5)
        p_value = binom_dist.cdf(min(tree_corr_bayes_wrong, bayes_corr_tree_wrong)) + (1-binom_dist.cdf(max(tree_corr_bayes_wrong, bayes_corr_tree_wrong) -1))
        print('The p-value is {:.100f}'.format(p_value))"""



        



    









    