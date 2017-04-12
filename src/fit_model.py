import joblib
import numpy as np
import tensorflow as tf
from definitions import ALL_CLASSES
from keras.layers import Dense, Embedding
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier

from src.preprocessing import norm_names, get_bag_of_word


def main():

    # all_data = joblib.load('small_data')
    # X_train = joblib.load('small_train')
    # X_test = joblib.load('small_test')
    # all_data = joblib.load('medium_data')
    # X_train = joblib.load('medium_train')
    # X_test = joblib.load('small_test')
    # all_data = joblib.load('all_data')
    # X_train = joblib.load('all_train')
    # X_test = joblib.load('all_test')
    all_data = joblib.load('all_data')
    X_train = joblib.load('tokenized_train')
    X_test = joblib.load('tokenized_test')
    y_train = all_data['y_train']
    y_test = all_data['y_test']
    keras(X_train,X_test,y_train,y_test)


def keras(X_train,X_test,y_train, y_test):


if __name__ == '__main__':
    main()
