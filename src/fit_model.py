import joblib
import numpy
from src.preprocessing import norm_names
from src.definitions import SMALL_DATA_PATH, SEQUENCE_LENGTH, WORD2VEC_DIM, FULL_DATA_PATH, FULL_W2V_X_TEST_PATH, FULL_W2V_X_TRAIN_PATH,\
    SMALL_W2V_X_TRAIN_PATH, SMALL_W2V_X_TEST_PATH, MEDIUM_DATA_PATH, MEDIUM_W2V_X_TRAIN_PATH, MEDIUM_W2V_X_TEST_PATH,ALL_CLASSES,\
    FULL_TFIDF_TEST_PATH, FULL_TFIDF_TRAIN_PATH
from sklearn.metrics import classification_report, log_loss
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
import keras
from keras import backend
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.regularizers import l2
import tensorflow as tf


class LossHistory(keras.callbacks.Callback):
    C = 950
    losses = []

    def some_shit(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def on_batch_end(self, batch, logs=None):
        if self.C % 950 == 0:
            y_pred = self.model.predict_classes(self.X_test[:128])
            print(y_pred)
            try:
                print(log_loss(self.y_test[:128], y_pred, labels=ALL_CLASSES))
                self.losses.append(log_loss(self.y_test[:128], y_pred, labels=ALL_CLASSES))
            except ValueError:
                pass
            print(classification_report(self.y_test[:128], y_pred, target_names=ALL_CLASSES))
        self.C += 95


def main():
    all_data = joblib.load(FULL_DATA_PATH)
    y_train = to_categorical(norm_names(all_data['y_train']),19)
    y_test = norm_names(all_data['y_test'])
    X_train = numpy.memmap(FULL_W2V_X_TRAIN_PATH, dtype='float64', mode='r', shape=(len(y_train), SEQUENCE_LENGTH*WORD2VEC_DIM))
    X_test = numpy.memmap(FULL_W2V_X_TEST_PATH,dtype='float64', mode='r', shape=(len(y_test), SEQUENCE_LENGTH*WORD2VEC_DIM))
    # X_train = joblib.load(FULL_TFIDF_TRAIN_PATH)
    # X_test = joblib.load(FULL_TFIDF_TEST_PATH)
    # model = LogisticRegressionCV(n_jobs=1,solver='sag', multi_class='multinomial', random_state=123, verbose=100000, max_iter=100000000)
    # model.fit(X_train,y_train)
    # y_pred = model.predict(X_test)
    # print(classification_report(y_test,y_pred))
    # joblib.dump(model, 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\models\\sklearn_log_reg_tf-idf.pkl')
    #tf_log(X_train,X_test, y_train,y_test)
    NBOW(X_train, X_test, y_train, y_test)


def NBOW(X_train, X_test, y_train, y_test):
    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            backend.set_session(sess)
            model = Sequential()
            opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
            history = LossHistory()
            history.some_shit(X_test, y_test)
            model.add(Dropout(0.8, input_shape=(SEQUENCE_LENGTH*WORD2VEC_DIM, )))
            model.add(Dense(512, bias_regularizer=l2(0.00001), activity_regularizer=l2(0.00001) ))
            model.add(Dense(19, activation='softmax'))
            model.compile(optimizer='adadelta', loss='categorical_crossentropy')
            model.fit(X_train, y_train, epochs=100, batch_size=95, callbacks=[history])
            y_pred = model.predict_classes(X_test)
            print(y_pred)
            print(classification_report(y_test,y_pred,target_names=ALL_CLASSES))


def tf_log(X_train, X_test, y_train, y_test):
    learning_rate = 0.001
    sess = tf.InteractiveSession()
    epoch_count = 10
    batch_size = 95
    input_tensor_dim = X_train.shape[1]
    output_tensor_dim = len(ALL_CLASSES)
    input_tensor = tf.placeholder(tf.float32, [None,input_tensor_dim])
    output_tensor = tf.placeholder(tf.float32, [None,output_tensor_dim])
    input_tensor_weights = tf.Variable(tf.zeros([input_tensor_dim,output_tensor_dim]))
    output_tensor_weights = tf.Variable(tf.zeros([output_tensor_dim]))
    model = tf.nn.softmax(tf.matmul(input_tensor, input_tensor_weights) + output_tensor_weights)
    cost = tf.reduce_mean(-tf.reduce_sum(output_tensor*tf.log(model), axis=1))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    y_train_tens = tf.one_hot(y_train, depth=output_tensor_dim).eval()
    init = tf.global_variables_initializer()
    kf = StratifiedKFold(n_splits=2,shuffle=True)
    for train, test in kf.split(numpy.zeros(len(y_train)), y_train):
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
                    _, c = sess.run([optimizer, cost], feed_dict={input_tensor: batch_in, output_tensor: batch_out})
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
                print(classification_report(numpy.asarray(y_train)[test], ress, target_names=ALL_CLASSES))
                if prev_acc > avg_acc:
                    break
                prev_acc = avg_acc

if __name__ == '__main__':
    main()
