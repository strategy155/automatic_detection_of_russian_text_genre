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

    #  naive_bayes(all_data)
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
    # slctr = SelectKBest(k=50000)
    # slctr.fit(X_train,all_data['y_train'])
    # X_train = slctr.transform(X_train)
    # X_test = slctr.transform(X_test)
    # tf_log(X_train,X_test,y_train,y_test)
    # (all_data)
    # mlp_clas(all_data)
    keras(X_train,X_test,y_train,y_test)
    # NBOW(X_train,X_test,y_train,y_test)



def NBOW(X_train,X_test,y_train,y_test):
    x = joblib.load('keras_tokenizer')
    print(x.word_counts)
    print(x.word_index)

def keras(X_train,X_test,y_train, y_test):
    embedding_vecor_length = 64
    top_words = 20000
    y_train = to_categorical(norm_names(y_train),19)
    y_test = to_categorical(norm_names(y_test),19)
    max_review_length = 2000
    model = Sequential()
    model.add(Embedding())
    model.add(Dense(19, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=64)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


def tf_log(X_train,X_test,y_train, y_test):
    learning_rate = 0.001
    sess = tf.InteractiveSession()
    epoch_count = 10
    batch_size = 1
    input_tensor_dim = X_train.shape[1]
    output_tensor_dim = len(ALL_CLASSES)
    input_tensor = tf.placeholder(tf.float32, [None,input_tensor_dim])
    output_tensor = tf.placeholder(tf.float32, [None,output_tensor_dim])
    input_tensor_weights = tf.Variable(tf.zeros([input_tensor_dim,output_tensor_dim]))
    output_tensor_weights = tf.Variable(tf.zeros([output_tensor_dim]))
    model = tf.nn.softmax(tf.matmul(input_tensor,input_tensor_weights) + output_tensor_weights)
    cost = tf.reduce_mean(-tf.reduce_sum(output_tensor*tf.log(model),axis=1))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    y_train_tens = tf.one_hot(norm_names(y_train), depth=output_tensor_dim).eval()
    init = tf.global_variables_initializer()
    kf = StratifiedKFold(n_splits=2,shuffle=True)
    for train,test in kf.split(X_train,y_train):
        with tf.Session() as sess:
            sess.run(init)
            prev_acc = 0
            for epoch in range(epoch_count):
                ress = []
                c = 0
                avg_acc = 0.
                train_batch_count = int(len(train) / batch_size)
                for i in range(train_batch_count):
                    batch_start = i*batch_size
                    batch_end = i*batch_size + batch_size
                    batch_in, batch_out = X_train[train[batch_start:batch_end]].toarray(), [y_train_tens[i] for i in train[batch_start:batch_end]]
                    _, c = sess.run([optimizer, cost], feed_dict={input_tensor: batch_in,output_tensor: batch_out})
                correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(output_tensor, 1))
                res = tf.argmax(model, 1)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                test_batch_count = int(len(test)/batch_size)
                for i in range(test_batch_count):
                    batch_start = i*batch_size
                    batch_end = i*batch_size + batch_size
                    ress += res.eval({input_tensor: X_train[test[batch_start:batch_end]].toarray(), output_tensor:[y_train_tens[i] for i in test[batch_start:batch_end]]}).tolist()
                    avg_acc += accuracy.eval({input_tensor: X_train[test[batch_start:batch_end]].toarray(), output_tensor:[y_train_tens[i] for i in test[batch_start:batch_end]]})
                avg_acc = avg_acc / test_batch_count
                print(classification_report(np.asarray(norm_names(y_train))[test], ress, target_names=ALL_CLASSES))
                if prev_acc > avg_acc:
                    break
                prev_acc = avg_acc


def fn_train(X_train,y_train):
    print(X_train.shape)
    indices = []
    for idx,X in enumerate(X_train):
        indices+=[[idx,i] for i in X.indices[::-1]]
    print(np.asarray(indices[0]))
    cat_cols = {
        'words': tf.SparseTensor(
            indices=indices,
            values=X_train.data,
            dense_shape=[X_train.shape[0], X_train.shape[1]]
        )
    }
    y_train = tf.constant(y_train)
    return cat_cols, y_train


def fn_test(X_train):
    print(X_train.shape)
    indices = []
    for idx,X in enumerate(X_train):
        indices+=[[idx,i] for i in X.indices[::-1]]
    print(np.asarray(indices[0]))
    cat_cols = {
        'words': tf.SparseTensor(
            indices=indices,
            values=X_train.data,
            dense_shape=[X_train.shape[0], X_train.shape[1]]
        )
    }
    return cat_cols



def mlp_clas(all_data):
    X_train = joblib.load('X_train')
    X_test = joblib.load('X_test')
    model = MLPClassifier(verbose=100000, activation='logistic', max_iter=300)
    model.fit(X_train, all_data['y_train'])
    y_pred = model.predict(X_test)
    print(classification_report(all_data['y_test'], y_pred))


def log_reg(all_data):
    X_train = joblib.load('small_train')
    X_test = joblib.load('small_test')
    model = LogisticRegressionCV(verbose=100,solver='sag',max_iter=100000,multi_class='ovr')
    model.fit(X_train, all_data['y_train'])
    y_pred = model.predict(X_test)
    print(classification_report(all_data['y_test'], y_pred))


def naive_bayes(all_data):
    batch_size = 100
    models = [
        GaussianNB(), MultinomialNB(), BernoulliNB()
    ]
    X_train = get_bag_of_word(all_data.X_train)
    X_test = get_bag_of_word(all_data.X_test)
    for model in models:
        for i in range(int(len(all_data.y_train)/batch_size)):
            X_batch = X_train[i*batch_size:(i+1)*batch_size,:].toarray()
            y_batch = all_data.y_train[(i)*batch_size:(i+1)*batch_size]
            model.partial_fit(X_batch, y_batch, classes=ALL_CLASSES)
        y_pred = model.predict(X_test.toarray())
        print(classification_report(all_data.y_test, y_pred))


if __name__ == '__main__':
    main()
