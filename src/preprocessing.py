import random
import re

import joblib
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer

from src.definitions import MAX_NUMBER, PREDICTION_DATA, ALL_CLASSES


def _choose_filenames(train_amount=MAX_NUMBER, test_amount=int(MAX_NUMBER / 10)):
    _genre_to_filenames = joblib.load('test_lul')
    _all_amount = train_amount + test_amount
    samples_dict = PREDICTION_DATA
    for key in _genre_to_filenames.keys():
        _file_amount = len(_genre_to_filenames[key])
        if _file_amount >= _all_amount:
            _range_gen = range(_file_amount)
            _random_indices = random.sample(_range_gen, train_amount+test_amount)
            _filenames = _genre_to_filenames[key]
            _fill_sample(train_amount,test_amount,key,_filenames,_random_indices,samples_dict)
    _shuffle_samples(samples_dict['X_train'],samples_dict['y_train'])
    _shuffle_samples(samples_dict['X_test'], samples_dict['y_test'])
    return samples_dict


def _fill_sample(train_amount, test_amount, key, filenames, random_indices, samples_dict):
    X_train, y_train = _fill_filenames_for_genre(train_amount, key, filenames, random_indices)
    X_test, y_test = _fill_filenames_for_genre(test_amount, key, filenames, random_indices, train_amount)
    samples_dict['X_train'] += X_train
    samples_dict['y_train'] += y_train
    samples_dict['X_test'] += X_test
    samples_dict['y_test'] += y_test
    return None


def _fill_filenames_for_genre(count, key, filenames, indices, start_number=0):
    X = []
    y = []
    for i in range(start_number,start_number + count):
        filename = filenames[indices[i]]
        X.append(filename)
        y.append(key)
    return X, y


def norm_names(answers):
    i=0
    _new_dic={}
    y_new = []
    for elem in ALL_CLASSES:
        _new_dic[elem] = i
        i+=1
    for elem in answers:
        y_new.append(_new_dic[elem])
    return y_new


def _shuffle_samples(X,y):
    meta = list(zip(X,y))
    random.shuffle(meta)
    X[:], y[:] = zip(*meta)
    return None


def _gen_docs(filenames):
    c = 0
    for filename in filenames:
        print(c, len(filenames))
        c+=1
        big_line = joblib.load(filename)
        yield big_line


def get_bag_of_word(filenames):
    try:
        tf_idf = joblib.load("tf_idf")
    except FileNotFoundError:
        tf_idf = TfidfVectorizer()
    X = tf_idf.transform(_gen_docs(filenames))
    return X


def _make_array(X):
    X_cool = []
    for elem in X:
         X_cool.append(elem.toarray())
    X_cool = numpy.asarray(X_cool)
    X_cool = X_cool.reshape((X_cool.shape[0],-1))
    return X_cool


def gen_bag_of_word(X, y, batch_size = 500):
    print(y.shape)
    while True:
        rng_state = numpy.random.get_state()
        index = numpy.arange(numpy.shape(X)[0])
        numpy.random.shuffle(index)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(y)
        new_X = []
        new_y = []
        i = 0
        for idx,elem in zip(y,X[index, :]):
            idx = numpy.array(idx)[numpy.newaxis]
            new_X.append(elem.toarray())
            new_y.append(idx)
            i += 1
            if i == batch_size:
                i=0
                yield (numpy.vstack(new_X), numpy.vstack(new_y))
                new_X = []
                new_y = []