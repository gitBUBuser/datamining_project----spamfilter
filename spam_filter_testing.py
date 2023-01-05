from statistics import mean, stdev
from sklearn import preprocessing
from spam_filter_classes import EmailAttributeList, EmailInfo
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from sklearn.naive_bayes import GaussianNB



if __name__ == "__main__":

    data = "AttributeData.ods"
    data_handler = EmailAttributeList()
    data_handler.load(data)
    
    X, y = data_handler.to_feature_vector()
    print(y)
    print(X)
    print(X.shape)
    skf = StratifiedKFold(n_splits=5)
    
    tree_clf = tree.DecisionTreeClassifier()
    bayes = GaussianNB()



    accuracy_scores_skf = []


    
    for train_index, test_index in skf.split(X, y):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        tree_clf.fit(x_train_fold, y_train_fold)
        accuracy_scores_skf.append(tree_clf.score(x_test_fold, y_test_fold))
    


    # Print the output.
    print('List of possible accuracy:', accuracy_scores_skf)
    print('\nMaximum Accuracy That can be obtained from this model is:',max(accuracy_scores_skf)*100, '%')
    print('\nMinimum Accuracy:', min(accuracy_scores_skf)*100, '%')
    print('\nOverall Accuracy:', mean(accuracy_scores_skf)*100, '%')
    print('\nStandard Deviation is:', stdev(accuracy_scores_skf))



    










