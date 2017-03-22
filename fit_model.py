import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import  GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
import collections
import random
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy
import json
from sklearn.model_selection import StratifiedKFold


CORPUS_PATH = 'D:\\usr\\gwm\\c_w\\lib.rus.ec'
ALL_CLASSES = ['sf','det','prose','love','adv','child','antique','sci','comp','ref','nonf','religi','humor','home']
MAX_NUMBER = 1000
PREDICTION_DATA ={'X_train':[], 'y_train':[], 'X_test':[], 'y_test':[]}







def _choose_filenames(train_amount = MAX_NUMBER, test_amount = int(MAX_NUMBER/10)):
    _genre_to_filenames = joblib.load('new_genre_list')
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


def _get_bag_of_word(filenames):
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


def _gen_bag_of_word(X,y,batch_size = 500):
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


def main():
    with open('all_samples','r') as f:
        all_data =json.loads(f.read())
    #  naive_bayes(all_data)
    log_reg(all_data)
    # mlp_clas(all_data)
    # keras(all_data)


def keras(all_data):
    model = Sequential()
    X_train = joblib.load('X_train')
    X_test = joblib.load('X_test')
    y_train = np_utils.to_categorical(norm_names(all_data['y_train']))
    y_test = np_utils.to_categorical(norm_names(all_data['y_test']))
    # slctr = SelectKBest(k=100)
    # slctr.fit(X_train,all_data['y_train'])
    # X_train = slctr.transform(X_train)
    # X_test = slctr.transform(X_test)
    model.add(Dense(64 , input_dim=X_test.shape[1], activation='softmax'))
    model.add(Dense(14, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit_generator(_gen_bag_of_word(X_train,y_train),samples_per_epoch=14000,verbose=2,nb_epoch=200)
    y_pred = model.evaluate(X_test,y_test,batch_size=32)
    y_what = model.predict_classes(X_test)
    print(y_pred,y_test,y_train,y_what)
    print(classification_report(norm_names(all_data['y_test']),y_what))


def mlp_clas(all_data):
    X_train = joblib.load('X_train')
    X_test = joblib.load('X_test')

    model = MLPClassifier(verbose=100000, activation='logistic', max_iter=300)
    model.fit(X_train, all_data['y_train'])
    y_pred = model.predict(X_test)
    print(classification_report(all_data['y_test'], y_pred))


def log_reg(all_data):
    X_train = joblib.load('X_train')
    X_test = joblib.load('X_test')

    model = LogisticRegressionCV(verbose=1000, n_jobs=-1, max_iter=10000)
    model.fit(X_train, all_data['y_train'])
    y_pred = model.predict(X_test)
    print(classification_report(all_data['y_test'], y_pred))


def naive_bayes(all_data):
    batch_size = 100
    models = [
        GaussianNB(), MultinomialNB(), BernoulliNB()
    ]
    X_train = _get_bag_of_word(all_data.X_train)
    X_test = _get_bag_of_word(all_data.X_test)
    for model in models:
        for i in range(int(len(all_data.y_train)/batch_size)):
            X_batch = X_train[i*batch_size:(i+1)*batch_size,:].toarray()
            y_batch = all_data.y_train[(i)*batch_size:(i+1)*batch_size]
            model.partial_fit(X_batch,y_batch, classes=ALL_CLASSES)
        y_pred = model.predict(X_test.toarray())
        print(classification_report(all_data.y_test, y_pred))


if __name__ == '__main__':
    main()
