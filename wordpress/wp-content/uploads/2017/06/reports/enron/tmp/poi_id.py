#!/usr/bin/python

import sys
import pickle

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
import numpy as np


class SelectAtMostKBest(SelectKBest):
    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            self.k = "all"


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# ALL of features_list.  We will end up using permutations on this to see which combination of features works best.
features_list = ['poi',
                 'bonus',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'from_poi_score',
                 'to_poi_score',
                 'salary',
                 'exercised_stock_options',
                 'total_stock_value',
                 'restricted_stock'
                 ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

## -- RESULTS --
# At this point the results are
# GaussianNB(priors=None)
#	Accuracy: 0.20660	Precision: 0.12419	Recall: 0.81800	F1: 0.21565	F2: 0.38634
#	Total predictions: 15000	True positives: 1636	False positives: 11537	False negatives:  364	True negatives: 1463


### Task 2: Remove outliers

# Now lets remove columns where we have a lot of missing data, lets say over 100 missing data points.  That leaves us removing
# columns : director_fees ( missing 129 ) , loan_advances ( 142 ), restricted_stock_deferred ( 128 ) , deferral_payments ( 107 )

missing_value_columns = ['director_fees', 'loan_advances', 'restricted_stock_deferred', 'deferral_payments']


def del_column(d, col_name):
    for key in d.keys():
        if col_name in d[key]:
            del d[key][col_name]
    return d


# Remove columns and the TOTAL row

for v in missing_value_columns:
    del_column(data_dict, v)
del data_dict['TOTAL']

## -- RESULTS --
# At this point the results are
# GaussianNB(priors=None)
#	Accuracy: 0.85467	Precision: 0.42188	Recall: 0.24300	F1: 0.30838	F2: 0.26552
#	Total predictions: 15000	True positives:  486	False positives:  666	False negatives: 1514	True negatives: 12334

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

for name in data_dict:
    from_poi_score = 0
    to_poi_score = 0
    if data_dict[name]['from_messages'] != 'NaN' and data_dict[name]['from_messages'] != 0:
        from_poi_score = float(data_dict[name]['from_this_person_to_poi']) / float(data_dict[name]['from_messages'])

    if data_dict[name]['to_messages'] != 'NaN' and data_dict[name]['to_messages'] == 0:
        to_poi_score = float(data_dict[name]['from_poi_to_this_person']) / float(data_dict[name]['to_messages'])

    data_dict[name]['from_poi_score'] = from_poi_score
    data_dict[name]['to_poi_score'] = to_poi_score

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
print(sum(labels))

# TASK 3.5 ,
CLF_ = DecisionTreeClassifier()

import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=CLF_, step=1, cv=StratifiedKFold(labels, 50),
          scoring='precision')
rfecv.fit(features, labels)
print("Optimal number of features : %d" % rfecv.n_features_)
print rfecv.support_
for pair in zip(features_list[1:], rfecv.support_):
    print(pair)
#features=features[:,rfecv.support_]
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# ### Task 4: Try a varity of classifiers
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 1
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
gnb = GaussianNB()
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
ada = AdaBoostClassifier(random_state=RANDOM_STATE)
classifiers = [dt, gnb, mlp, ada]

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)

param_grid = [
    {
        "pca__n_components": range(1, len(features_list) - 1, 1),
        "selectatmostkbest__k": range(1, len(features_list) - 1, 1),
        "decisiontreeclassifier__criterion": ['gini', 'entropy'],
        "decisiontreeclassifier__splitter": ['best', 'random'],
        "decisiontreeclassifier__presort": [True, False]

    },
    {
        "pca__n_components": range(1, len(features_list) - 1, 1),
        "selectatmostkbest__k": range(1, len(features_list) - 1, 1),
    },
    {
        "selectatmostkbest__k": range(1, len(features_list) - 1, 1),
        "pca__n_components": range(1, len(features_list) - 1, 1),
        "mlpclassifier__hidden_layer_sizes": [(5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7)]
    },
    {
        "selectatmostkbest__k": range(1, len(features_list) - 1, 1),
        "pca__n_components": range(1, len(features_list) - 1, 1),
        "adaboostclassifier__learning_rate": [0.1, 1, 10],
        "adaboostclassifier__algorithm": ['SAMME', 'SAMME.R']

    }
]

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


scores = ['precision']
for score in scores:
    INDEX = 0
    for c in classifiers:
        params = param_grid[INDEX]
        clf = c

        pipe = make_pipeline(StandardScaler(), PCA(n_components=len(features_list) - 1),
                             SelectAtMostKBest(k=len(features_list) - 1), c)
        search = GridSearchCV(pipe, params, cv=StratifiedShuffleSplit(), scoring=score, n_jobs=-1)
        search.fit(features_train, labels_train)
        search.score(features_test, labels_test)

        print("BEST SCORE = {}, CLF = {}".format(str(search.best_score_), clf))
        print("BEST PARAMS = {} ".format(search.best_params_))
        print("BEST ESTIMATOR = {}".format(search.best_estimator_))
        INDEX += 1

        # Break on GuassianNB, the best predictor for testing
        if INDEX == 2:
            break

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
