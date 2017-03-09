import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import  GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import classification_report
import collections
import random
from keras.models import Sequential
from keras.layers import Dense,Activation
import numpy
import scipy.sparse

CORPUS_PATH = 'D:\\usr\\gwm\\c_w\\lib.rus.ec'
ALL_CLASSES = ['sf','det','prose','love','adv','child','antique','sci','comp','ref','nonf','religi','humor','home']

def _choose_filenames(max_number = 1000):
    data = joblib.load('new_genre_list')
    X_train_filenames = []
    X_test_filenames = []
    y_train = []
    y_test = []
    c = 0
    for key in data.keys():
        if len(data[key]) > 1000:
            rnd_list = random.sample(range(len(data[key])), max_number)
            for i in range(int(9*max_number/10)):
                X_train_filenames.append(data[key][rnd_list[i]])
                y_train.append(key)
            for i in range(int(9*max_number/10), max_number):
                X_test_filenames.append(data[key][rnd_list[i]])
                y_test.append(key)
    X_train_filenames, y_train = _shuffle_samples(X_train_filenames,y_train)
    X_test_filenames, y_test = _shuffle_samples(X_test_filenames, y_test)
    prediction_data = collections.namedtuple('Dataset',['X_train','y_train','X_test','y_test'])
    genre_data = prediction_data(X_train=X_train_filenames,y_train=y_train,X_test=X_test_filenames,y_test=y_test)
    return genre_data


def norm_names(y):
    i=0
    new_dic={}
    y_new = []
    for elem in ALL_CLASSES:
        new_dic[elem] = i
        i+=1
    for elem in y:
        y_new.append(new_dic[elem])
    return y_new


def _shuffle_samples(X,y):
    meta = list(zip(X,y))
    random.shuffle(meta)
    X[:], y[:] = zip(*meta)
    return X, y

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
    return X.todense()





def main():
    all_data = _choose_filenames()
    #  naive_bayes(all_data)
    # log_reg(all_data)
    # mlp_clas(all_data)
    keras(all_data)


def keras(all_data):
    model = Sequential()
    X_train = _get_bag_of_word(all_data.X_train)
    X_test = _get_bag_of_word(all_data.X_test)
    y_train = norm_names(all_data.y_train)
    y_test = norm_names(all_data.y_test)
    model.add(Dense(128 ,input_dim=X_train.shape[1],activation='softmax'))
    model.add(Dense(1))
    # chunk=numpy.asarray(list(chunks(X_train)))
    # print(chunk)
    model.compile(optimizer='sgd',loss='mse')
    model.fit(X_train, y_train, verbose=2,batch_size=1)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


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
    print(classification_report(all_data.y_test, y_pred))


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
