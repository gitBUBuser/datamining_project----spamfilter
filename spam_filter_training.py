from statistics import mean, stdev
import numpy as np
import os
from sklearn import preprocessing
from spam_filter_classes import EmailAttributeList, EmailInfo
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from itertools import compress, product
import time
import io
from collections import OrderedDict
import matplotlib.pyplot as plt

from math import ceil
from math import sqrt

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def plot_tree_depth_errors_all(attribute_errors):

    depths = []
    attributes_titles = list(attribute_errors.keys())


    train_errors = []
    test_errors = []

    for attributes, errors in attribute_errors.items():
        t_depths = []
        t_train_errors = []
        t_test_errors = []

        for depth, mean_errors in errors.items():
            t_depths.append(depth)
            t_train_errors.append(mean_errors["train"])
            t_test_errors.append(mean_errors["test"])

        depths.append(t_depths)
        train_errors.append(t_train_errors)
        test_errors.append(t_test_errors)
    
    plt_axis = ceil(sqrt(len(attributes)))

    fig, ax = plt.subplots(plt_axis, plt_axis, sharex=True, sharey=True, figsize=(20, 20), constrained_layout = True)
    fig.suptitle('generalization error vs training error by depth', fontsize=12)
    for i in range(plt_axis):
        for j in range(plt_axis):
            if i + j < len(attributes_titles):
                test = i + j
                ax[i][j].set_title(f"{attributes_titles[i + j]}.", fontsize=3)
                ax[i][j].plot(depths[i + j], train_errors[i + j], label = "training errors")
                ax[i][j].plot(depths[i + j], test_errors[i + j], label = "generalization_errors")
                ax[i][j].set_xlabel("tree depth")
                ax[i][j].set_ylabel("error rate")
            else:
                break
    plt.legend()
    plt.show()

def plot_tree_depth_weighted_accuracy_all(attribute_accuracies):

    depths = []
    attributes_titles = list(attribute_accuracies.keys())
    train_accuracies = []
    test_accuracies = []

    for attributes, accuracy in attribute_accuracies.items():
        t_depths = []
        t_train_accuracies = []
        t_test_accuracies = []

        for depth, mean_accuracy in accuracy.items():
            t_depths.append(depth)
            t_train_accuracies.append(mean_accuracy["train"])
            t_test_accuracies.append(mean_accuracy["test"])

        depths.append(t_depths)
        train_accuracies.append(t_train_accuracies)
        test_accuracies.append(t_test_accuracies)
    
    plt_axis = ceil(sqrt(len(attributes)))

    fig, ax = plt.subplots(plt_axis, plt_axis, sharex=True, sharey=True, figsize=(20, 20), constrained_layout = True)
    fig.suptitle('generalization error vs training accuracy by depth(favoring ham -> ham (5-1))', fontsize=12)
    for i in range(plt_axis):
        for j in range(plt_axis):
            if i + j < len(attributes_titles):
                test = i + j
                ax[i][j].set_title(f"{attributes_titles[i + j]}.", fontsize=3)
                ax[i][j].plot(depths[i + j], train_accuracies[i + j], label = "training accuracy")
                ax[i][j].plot(depths[i + j], test_accuracies[i + j], label = "generalization accuracy")
                ax[i][j].set_xlabel("tree depth")
                ax[i][j].set_ylabel("weighted accuracy")
            else:
                break
    plt.legend()
    plt.show()

def plot_bayes_smoothing_errors_all(attribute_errors):
    smoothings = []
    attributes_titles = list(attribute_errors.keys())

    train_errors = []
    test_errors = []

    for attributes, errors in attribute_errors.items():
        t_smoothings = []
        t_train_errors = []
        t_test_errors = []

        for smoothing, mean_errors in errors.items():
            t_smoothings.append(smoothing)
            t_train_errors.append(mean_errors["train"])
            t_test_errors.append(mean_errors["test"])

        smoothings.append(t_smoothings)
        train_errors.append(t_train_errors)
        test_errors.append(t_test_errors)
    
    plt_axis = ceil(sqrt(len(attributes)))

    fig, ax = plt.subplots(plt_axis, plt_axis, sharex=True, sharey=True, figsize=(20, 20), constrained_layout = True)
    fig.suptitle('generalization error vs training error by smoothing', fontsize=12)
    for i in range(plt_axis):
        for j in range(plt_axis):
            if i + j < len(attributes_titles):
                test = i + j
                ax[i][j].set_title(f"{attributes_titles[i + j]}.", fontsize=3)
                ax[i][j].plot(smoothings[i + j], train_errors[i + j], label = "training errors")
                ax[i][j].plot(smoothings[i + j], test_errors[i + j], label = "generalization_errors")
                ax[i][j].set_xlabel("bayes smoothing")
                ax[i][j].set_ylabel("error rate")
            else:
                break

    plt.legend()
    plt.show()

def plot_bayes_smoothing_accuracy_all(attribute_errors):
    smoothings = []
    attributes_titles = list(attribute_errors.keys())

    train_errors = []
    test_errors = []

    for attributes, errors in attribute_errors.items():
        t_smoothings = []
        t_train_errors = []
        t_test_errors = []

        for smoothing, mean_errors in errors.items():
            t_smoothings.append(smoothing)
            t_train_errors.append(mean_errors["train"])
            t_test_errors.append(mean_errors["test"])

        smoothings.append(t_smoothings)
        train_errors.append(t_train_errors)
        test_errors.append(t_test_errors)
    
    plt_axis = ceil(sqrt(len(attributes)))

    fig, ax = plt.subplots(plt_axis, plt_axis, sharex=True, sharey=True, figsize=(20, 20), constrained_layout = True)
    fig.suptitle('generalization accuracy vs training accuracy by smoothing (favoring ham -> ham (5-1))', fontsize=12)
    for i in range(plt_axis):
        for j in range(plt_axis):
            if i + j < len(attributes_titles):
                test = i + j
                ax[i][j].set_title(f"{attributes_titles[i + j]}.", fontsize=3)
                ax[i][j].plot(smoothings[i + j], train_errors[i + j], label = "training accuracy")
                ax[i][j].plot(smoothings[i + j], test_errors[i + j], label = "generalization accuracy")
                ax[i][j].set_xlabel("bayes smoothing")
                ax[i][j].set_ylabel("weighted accuracy")
            else:
                break

    plt.legend()
    plt.show()


def total_cost_ratio(cost_dict, Ham_To_Spam_w = 1):
    n_spam = cost_dict["FP"] + cost_dict["TN"]
    divider = (Ham_To_Spam_w * cost_dict["FN"]) + cost_dict["FP"]
    return n_spam / divider

def accuracy(cost_dict, Ham_To_Ham_w = 1):
    return (cost_dict["TP"] +  (Ham_To_Ham_w * cost_dict["TN"])) / (cost_dict["FP"] + cost_dict["FN"] + cost_dict["TP"] + cost_dict["TN"])

def precision(cost_dict):
    return cost_dict["TP"] / (cost_dict["TP"] + cost_dict["FP"])

def recall(cost_dict):
    return cost_dict["TP"] / (cost_dict["TP"] + cost_dict["FN"])

def f1(cost_dict):
    prec = precision(cost_dict)
    rec = recall(cost_dict)
    return (2 * rec * prec) / (prec + rec)


def get_cost_table(y_pred, y_true):
    table = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                table["TP"] += 1
            else:
                table["FP"] += 1
        else:
            if y_true[i] == 0:
                table["TN"] += 1
            else:
                table["FN"] += 1
    return table



if __name__ == "__main__":
    data = "AttributeData.ods"
    data_handler = EmailAttributeList()
    data_handler.load(data)
    data_handler.restrict_to_amount_per_class(500)
    print(len(data_handler.get_emails()))
    tree_depths = np.arange(2, 40, 1)
    tree_splits = []

    best_attributes_tree = [
        ('LIX', 'Unknown_tags_relative', 'NOUN_relative', 'NUM_relative', 'ADV_relative', 'PRON_relative', 'DET_relative', 'PROPN_relative', 'SYM_relative', 'INTJ_relative'),
        ('SCONJ_relative', 'LIX', 'Unknown_tags_relative', 'NOUN_relative', 'VERB_relative', 'NUM_relative', 'ADV_relative', 'PRON_relative', 'CCONJ_relative', 'ADJ_relative', 'polarity', 'capital_words_relative', 'INTJ_relative'),
        ('LIX', 'Unknown_tags_relative', 'VERB_relative', 'AUX_relative', 'PRON_relative', 'DET_relative', 'CCONJ_relative', 'polarity', 'capital_words_relative', 'SYM_relative'),
        ('SCONJ_relative', 'LIX', 'Unknown_tags_relative', 'PART_relative', 'NUM_relative', 'PRON_relative', 'DET_relative', 'polarity', 'ADP_relative', 'INTJ_relative'),
        ('SCONJ_relative', 'LIX', 'NOUN_relative', 'ADV_relative', 'polarity', 'SYM_relative', 'ADP_relative', 'Unknown_tags_relative'),
        ('SCONJ_relative', 'LIX', 'Unknown_tags_relative', 'NOUN_relative', 'NUM_relative', 'ADV_relative', 'DET_relative', 'CCONJ_relative', 'PROPN_relative', 'polarity', 'capital_words_relative', 'INTJ_relative'),
        ('SCONJ_relative', 'LIX', 'Unknown_tags_relative', 'PART_relative', 'NOUN_relative', 'VERB_relative', 'ADV_relative', 'AUX_relative', 'PRON_relative', 'DET_relative', 'polarity', 'capital_words_relative', 'ADP_relative', 'SYM_relative'),
        ('SCONJ_relative', 'LIX', 'Unknown_tags_relative', 'PART_relative', 'VERB_relative', 'NUM_relative', 'ADV_relative', 'AUX_relative', 'PRON_relative', 'CCONJ_relative', 'PROPN_relative', 'polarity', 'capital_words_relative', 'ADP_relative', 'SYM_relative'),
        ('SCONJ_relative', 'LIX', 'PART_relative', 'NOUN_relative', 'AUX_relative', 'PRON_relative', 'DET_relative', 'CCONJ_relative', 'ADJ_relative', 'PROPN_relative', 'Unknown_tags_relative'),
        ('SCONJ_relative', 'Unknown_tags_relative', 'INTJ_relative', 'VERB_relative', 'ADV_relative', 'PRON_relative', 'CCONJ_relative', 'ADJ_relative', 'polarity', 'capital_words_relative', 'ADP_relative', 'SYM_relative')
    ]

    best_attributes_bayes = [
        ('LIX', 'NOUN_relative', 'ADV_relative', 'DET_relative', 'polarity', 'SYM_relative'),
        ('SCONJ_relative', 'NOUN_relative', 'AUX_relative', 'DET_relative', 'polarity', 'capital_words_relative', 'INTJ_relative'),
        ('SCONJ_relative', 'NOUN_relative', 'PRON_relative', 'DET_relative', 'CCONJ_relative', 'ADJ_relative', 'PROPN_relative', 'polarity'),
        ('SCONJ_relative', 'LIX', 'NOUN_relative', 'AUX_relative', 'DET_relative', 'CCONJ_relative', 'polarity', 'INTJ_relative'),
        ('LIX', 'Unknown_tags_relative', 'PART_relative', 'NOUN_relative', 'VERB_relative', 'AUX_relative', 'DET_relative', 'ADJ_relative', 'capital_words_relative', 'ADP_relative', 'SYM_relative'),
        ('LIX', 'Unknown_tags_relative', 'PART_relative', 'VERB_relative', 'ADV_relative', 'PRON_relative', 'PROPN_relative', 'ADJ_relative', 'capital_words_relative', 'ADP_relative', 'SYM_relative'),
        ('SCONJ_relative', 'LIX', 'Unknown_tags_relative', 'INTJ_relative', 'PART_relative', 'DET_relative', 'PROPN_relative', 'ADJ_relative', 'polarity', 'capital_words_relative', 'ADP_relative', 'SYM_relative'),
        ('SCONJ_relative', 'PART_relative', 'NOUN_relative', 'ADV_relative', 'PRON_relative', 'DET_relative', 'CCONJ_relative', 'polarity'),
        ('Unknown_tags_relative', 'NOUN_relative', 'AUX_relative', 'PRON_relative', 'DET_relative', 'CCONJ_relative', 'ADJ_relative', 'PROPN_relative', 'capital_words_relative', 'INTJ_relative'),
        ('SCONJ_relative', 'LIX', 'Unknown_tags_relative', 'INTJ_relative', 'VERB_relative', 'ADV_relative', 'DET_relative', 'CCONJ_relative', 'PROPN_relative', 'ADJ_relative', 'capital_words_relative', 'SYM_relative')
    ]

    best_attributes = best_attributes_tree + best_attributes_bayes

    #removes duplicates
    best_attributes = list(dict.fromkeys(best_attributes))


    attribute_errors = {}
    attribute_w_accuracy = {}

    best_tree_scores = np.zeros(3)
    best_tree_attributes = np.zeros(3,dtype=object)
    best_tree_depths = np.zeros(3)

    
    for attributes in best_attributes:
        X, y = data_handler.to_feature_vector(attributes)    
        skf = StratifiedKFold(n_splits=15)

        depth_errors = {}
        depth_accuracys = {}
        #pruning the tree
        for depth in tree_depths:
            tree_train_test_errors = {}
            tree_train_test_accuracy = {}

            tree_train_errors = []
            tree_test_errors = []

            tree_train_accuracy = []
            tree_test_accuracy = []

            tree_clf = DecisionTreeClassifier(max_depth=depth, splitter="best")
            for train_index, test_index in skf.split(X, y):
                x_train_fold, x_test_fold = X[train_index], X[test_index]
                y_train_fold, y_test_fold = y[train_index], y[test_index]

                tree_clf.fit(x_train_fold, y_train_fold)

                cost_table_test_tree = get_cost_table(tree_clf.predict(x_test_fold), y_test_fold)
                cost_table_train_tree = get_cost_table(tree_clf.predict(x_train_fold), y_train_fold)

                tree_train_errors.append(1 - accuracy(cost_table_train_tree))
                tree_test_errors.append(1 - accuracy(cost_table_test_tree))

                tree_train_accuracy.append(accuracy(cost_table_train_tree, Ham_To_Ham_w=5))
                tree_test_accuracy.append(accuracy(cost_table_test_tree, Ham_To_Ham_w=5))
                
            tree_train_test_errors["train"] = mean(tree_train_errors)
            tree_train_test_errors["test"] = mean(tree_test_errors)

            tree_train_test_accuracy["train"] = mean(tree_train_accuracy)
            tree_train_test_accuracy["test"] = mean(tree_test_accuracy)

            if mean(tree_test_accuracy) > best_tree_scores[0]:
                best_tree_scores[0] = mean(tree_test_accuracy)
                best_tree_attributes[0] = attributes
                best_tree_depths[0] = depth
                accuracy_sort = np.argsort(best_tree_scores)
                best_tree_scores = best_tree_scores[accuracy_sort]
                best_tree_attributes = best_tree_attributes[accuracy_sort]
                best_tree_depths = best_tree_depths[accuracy_sort]


            depth_accuracys[depth] = tree_train_test_accuracy
            depth_errors[depth] = tree_train_test_errors
         
        attribute_w_accuracy[attributes] = depth_accuracys
        attribute_errors[attributes] = depth_errors
    plot_tree_depth_weighted_accuracy_all(attribute_w_accuracy)
    plot_tree_deptj_errors_all(attribute_errors)
    print(best_tree_depths)
    print(best_tree_scores)
    print(best_tree_attributes)
 #   plot_tree_depth_errors_all(attribute_errors)
    
    low_smooth = 0.0000000001
    hi_smooth = 0.00000001
    diff = hi_smooth - low_smooth
    print(diff)
    step = diff / 50
    
    best_bayes_scores = np.zeros(3)
    best_bayes_attributes = np.zeros(3,dtype=object)
    best_bayes_depths = np.zeros(3)

    wanted_smoothings = np.arange(low_smooth, hi_smooth, step)
    print(wanted_smoothings)
    attribute_errors = {}
    attribute_accuracies = {}
    for attributes in best_attributes:
        X, y = data_handler.to_feature_vector(attributes)    
        skf = StratifiedKFold(n_splits=15)

        smoothing_errors = {}
        smoothing_accuracy = {}
        
        for smoothing in wanted_smoothings:
            bayes_clf = GaussianNB(var_smoothing=smoothing)
            bayes_train_test_errors = {}
            bayes_train_errors = []
            bayes_test_errors = []

            bayes_train_test_accuracy = {}
            bayes_train_accuracy = []
            bayes_test_accuracy = []

            for train_index, test_index in skf.split(X, y):
                x_train_fold, x_test_fold = X[train_index], X[test_index]
                y_train_fold, y_test_fold = y[train_index], y[test_index]

                bayes_clf.fit(x_train_fold, y_train_fold)

                cost_table_test_bayes = get_cost_table(bayes_clf.predict(x_test_fold), y_test_fold)
                cost_table_train_bayes = get_cost_table(bayes_clf.predict(x_train_fold), y_train_fold)
                bayes_train_errors.append(1 - accuracy(cost_table_train_bayes))
                bayes_test_errors.append(1 - accuracy(cost_table_test_bayes))
                bayes_train_accuracy.append(accuracy(cost_table_train_bayes, Ham_To_Ham_w=5))
                bayes_test_accuracy.append(accuracy(cost_table_test_bayes, Ham_To_Ham_w=5))
            
            bayes_train_test_errors["train"] = mean(bayes_train_errors)
            bayes_train_test_errors["test"] = mean(bayes_test_errors)
            bayes_train_test_accuracy["train"] = mean(bayes_train_accuracy)
            bayes_train_test_accuracy["test"] = mean(bayes_test_accuracy)

            if mean(bayes_test_accuracy) > best_bayes_scores[0]:
                best_bayes_scores[0] = mean(bayes_test_accuracy)
                best_bayes_attributes[0] = attributes
                best_bayes_depths[0] = smoothing
                accuracy_sort = np.argsort(best_bayes_scores)
                best_bayes_scores = best_bayes_scores[accuracy_sort]
                best_bayes_attributes = best_bayes_attributes[accuracy_sort]
                best_bayes_depths = best_bayes_depths[accuracy_sort]

            smoothing_errors[smoothing] = bayes_train_test_errors
            smoothing_accuracy[smoothing] = bayes_train_test_accuracy

        attribute_accuracies[attributes] = smoothing_accuracy
        attribute_errors[attributes] = smoothing_errors


    plot_bayes_smoothing_accuracy_all(attribute_accuracies)
    plot_bayes_smoothing_errors_all(attribute_errors)
    print()
    print(best_bayes_depths)
    print(best_bayes_scores)
    print(best_bayes_attributes)












