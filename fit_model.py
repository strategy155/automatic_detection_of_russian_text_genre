import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import  GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import classification_report
import collections
import random
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
import numpy
import scipy.sparse


CORPUS_PATH = 'D:\\usr\\gwm\\c_w\\lib.rus.ec'
ALL_CLASSES = ['sf','det','prose','love','adv','child','antique','sci','comp','ref','nonf','religi','humor','home']
MAX_NUMBER = 100


def _init_named_tuple():
    prediction_data = collections.namedtuple('Dataset',['X_train','y_train','X_test','y_test'])
    genre_data = prediction_data
    return genre_data


def _choose_filenames(train_amount = MAX_NUMBER, test_amount = int(MAX_NUMBER/10)):
    _genre_to_filenames = joblib.load('new_genre_list')
    _samples_tuple = _init_named_tuple()
    _all_amount = train_amount + test_amount
    for key in _genre_to_filenames.keys():
        _file_amount = len(_genre_to_filenames[key])
        if _file_amount >= _all_amount:
            _range_gen = range(_file_amount)
            _random_indices = random.sample(_range_gen, train_amount+test_amount)
            _filenames = _genre_to_filenames[key]
            try :
                X_train,y_train =_fill_filenames_for_genre(train_amount, key, _filenames, _random_indices)
                X_test, y_test = _fill_filenames_for_genre(test_amount, key, _filenames, _random_indices, train_amount)
                _samples_tuple.X_train+= X_train
                _samples_tuple.y_train +=y_train
                _samples_tuple.X_test += X_test
                _samples_tuple.y_test += y_test
            except TypeError:
                _samples_tuple.X_train, _samples_tuple.y_train = _fill_filenames_for_genre(train_amount,key, _filenames, _random_indices)
                _samples_tuple.X_test, _samples_tuple.y_test = _fill_filenames_for_genre(test_amount, key, _filenames, _random_indices, train_amount)
    _shuffle_samples(_samples_tuple.X_train,_samples_tuple.y_train)
    _shuffle_samples(_samples_tuple.X_test, _samples_tuple.y_test)
    return _samples_tuple


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


def _gen_bag_of_word(X,y):
    print(y.shape)
    while True:
        rng_state = numpy.random.get_state()
        index = numpy.arange(numpy.shape(X)[0])
        numpy.random.shuffle(index)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(y)
        for idx,elem in zip(y,X[index, :]):
            idx = numpy.array(idx)[numpy.newaxis]
            yield (elem.toarray(), idx)

def main():
    all_data = _choose_filenames()
    #  naive_bayes(all_data)
    # log_reg(all_data)
    # mlp_clas(all_data)
    keras(all_data)


def keras(all_data):
    model = Sequential()
    X_train = _get_bag_of_word(all_data.X_train)
    X_test = _make_array(_get_bag_of_word(all_data.X_test))
    y_train = np_utils.to_categorical(norm_names(all_data.y_train))
    y_test = np_utils.to_categorical(norm_names(all_data.y_test))
    print(y_train)
    print(X_train)
    model.add(Dense(32,input_dim=X_test.shape[1],activation='sigmoid'))
    model.add(Dense(14,activation='sigmoid'))
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    model.fit_generator(_gen_bag_of_word(X_train,y_train),samples_per_epoch=140,verbose=2,nb_epoch=20)
    y_pred = model.evaluate(X_test,y_test,batch_size=32)
    y_what = model.predict_classes(X_test)
    print(y_pred,y_test,y_train,y_what)
    print(classification_report(norm_names(all_data.y_test),y_what))

def chunks(array):
    for i in range(0,array.shape[1], 1000):
        try:
            yield (array[:, i:i+1000].toarray())
        except IndexError:
            continue

def mlp_clas(all_data):
    model = MLPClassifier(verbose=100000, activation='logistic', max_iter=300)
    model.fit(_get_bag_of_word(all_data.X_train), all_data.y_train)
    y_pred = model.predict(_get_bag_of_word(all_data.X_test))
    print(classification_report(all_data.y_test, y_pred))


def log_reg(all_data):
    model = LogisticRegressionCV(verbose=1000, n_jobs=1, max_iter=10000)
    model.fit(_get_bag_of_word(all_data.X_train), all_data.y_train)
    y_pred = model.predict(_get_bag_of_word(all_data.X_test))
    print(classification_report(all_data.y_test, y_pred,target_names=ALL_CLASSES))


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
