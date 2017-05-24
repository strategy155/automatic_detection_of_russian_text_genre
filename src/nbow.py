import joblib
import numpy
from src.preprocessing import norm_names
from src.definitions import SMALL_DATA_PATH, SEQUENCE_LENGTH, WORD2VEC_DIM, FULL_DATA_PATH, FULL_W2V_X_TEST_PATH, FULL_W2V_X_TRAIN_PATH,\
    SMALL_W2V_X_TRAIN_PATH, SMALL_W2V_X_TEST_PATH, MEDIUM_DATA_PATH, MEDIUM_W2V_X_TRAIN_PATH, MEDIUM_W2V_X_TEST_PATH,ALL_CLASSES,\
    FULL_TFIDF_TEST_PATH, FULL_TFIDF_TRAIN_PATH
import keras



def main():
    all_data = joblib.load(FULL_DATA_PATH)
    y_train = norm_names(all_data['y_train'])
    y_test = norm_names(all_data['y_test'])
    X_train = numpy.memmap(FULL_W2V_X_TRAIN_PATH, dtype='float64', mode='r', shape=(1900, WORD2VEC_DIM))
    X_test = numpy.memmap(FULL_W2V_X_TEST_PATH,dtype='float64', mode='r', shape=(len(y_test), WORD2VEC_DIM))
    prd = DAN()
    prd.train(X_train, X_test, y_train, y_test)


class DAN(object):
    BATCH_SIZE = 1024

    def __init__(self):
        self.train_data = None

    def train(self, train_data, test_data, train_target, test_target):
        self.train_data = train_data
        self.test_data = test_data
        self.train_target = keras.utils.to_categorical(train_target, len(ALL_CLASSES))
        self.test_target = keras.utils.to_categorical(test_target, len(ALL_CLASSES))
        self.model = keras.models.Sequential()
        print(train_data, train_target)
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
        self.model.add(keras.layers.Dense(300, input_shape=(self.train_data.shape[1],)))
        self.model.add(keras.layers.Dense(300))
        self.model.add(keras.layers.Dense(19, kernel_initializer=keras.initializers.RandomNormal(), bias_initializer=keras.initializers.Zeros(), activation='softmax', activity_regularizer=keras.regularizers.l2(0.00001), bias_regularizer=keras.regularizers.l2(0.00001), kernel_regularizer=keras.regularizers.l2(0.00001)))
        opt = keras.optimizers.Adadelta(epsilon=1e-06)
        self.model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        self.model.fit(self.train_data, self.train_target, epochs=500, validation_data=(self.test_data, self.test_target),  batch_size=32,
                       shuffle=True, callbacks=[earlyStopping], verbose=1)
        print(self.model.evaluate(self.test_data, self.test_target, verbose=0))


if __name__ == '__main__':
    main()
