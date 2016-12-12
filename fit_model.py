from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold, SelectKBest,chi2
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier,ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
import json
import numpy
GENRE_DATA_FILE='genre_data.json'


def get_genre_data(filename):
    with open(filename, 'r') as file:
        input_list_plus_list_of_answers = json.loads(file.read())
    return input_list_plus_list_of_answers


def get_redefined_answers(list_of_answers):
    new_list_of_answers = []
    for genre_code in list_of_answers:
        if genre_code.startswith('det') or genre_code == 'thriller':
            genre_code = 'det'
        elif genre_code.startswith('sf'):
            genre_code = 'sf'
        elif genre_code.startswith('prose') or genre_code.startswith('proce') or genre_code.startswith('short'):
            genre_code = 'prose'
        elif genre_code.startswith('love') or genre_code.startswith('romance'):
            genre_code = 'love'
        elif genre_code.startswith('adv'):
            genre_code = 'adv'
        elif genre_code == 'dramaturgy':
            genre_code = 'poetry'
        elif genre_code.startswith('antique'):
            genre_code = 'prose'
        elif genre_code.startswith('sci'):
            genre_code = 'sci'
        elif genre_code.startswith('comp'):
            genre_code = 'sci'
        elif genre_code.startswith('ref'):
            genre_code = 'prose'
        elif genre_code.startswith('nonf') or genre_code == 'design':
            genre_code = 'nonf'
        elif genre_code.startswith('religion'):
            genre_code = 'nonf'
        elif genre_code.startswith('humor'):
            genre_code = 'adv'
        elif genre_code.startswith('home'):
            genre_code = 'prose'
        elif genre_code.startswith('child'):
            genre_code = 'adv'
        new_list_of_answers.append(genre_code)
    return new_list_of_answers


def get_new_genre_features(new_answers, genre_features_and_answers):
    genre_features = {}
    for genre_code, features in zip(new_answers, genre_features_and_answers):
        try:
            genre_features[genre_code].append(features)
        except KeyError:
            genre_features[genre_code] = [features]
    return genre_features


def remove_not_enough_text_genre(genre_features):
    for genre_code in list(genre_features.keys()):
        print(genre_code, len(genre_features[genre_code]))
    return genre_features


genre_features_and_answers = get_genre_data(GENRE_DATA_FILE)
answers = genre_features_and_answers[-1]
del genre_features_and_answers[-1]
new_answers = get_redefined_answers(answers)
genre_features = get_new_genre_features(new_answers, genre_features_and_answers)
genre_features = remove_not_enough_text_genre(genre_features)
training_amount = 50
train_features = []
train_answers = []
test_features = []
test_answers = []
for genre_code in genre_features.keys():
    for j in range(0, int(len(genre_features[genre_code]) / 2)):
        train_features.append(genre_features[genre_code][j])
        train_answers.append(genre_code)
    for i in range(int(len(genre_features[genre_code]) / 2), len(genre_features[genre_code])):
        test_features.append(genre_features[genre_code][i])
        test_answers.append(genre_code)
training_array = numpy.array(train_features)
testing_array = numpy.array(test_features)


def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(training_array, train_answers)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(testing_array)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(test_answers, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
  ('classification', LinearSVC())
])))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()