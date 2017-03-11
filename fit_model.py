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

def _choose_filenames(max_number = 20):
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
        print(index)
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
    model.add(Dense(32,input_dim=X_test.shape[1]))
    model.add(Activation('sigmoid'))
    model.add(Dense(14,activation='sigmoid'))
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    model.fit_generator(_gen_bag_of_word(X_train,y_train),samples_per_epoch=252,verbose=2,nb_epoch=20)
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
