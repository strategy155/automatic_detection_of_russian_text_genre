import joblib
import numpy
from src.preprocessing import norm_names
from src.definitions import SMALL_DATA_PATH, SEQUENCE_LENGTH, WORD2VEC_DIM, FULL_DATA_PATH, FULL_W2V_X_TEST_PATH, FULL_W2V_X_TRAIN_PATH,\
    SMALL_W2V_X_TRAIN_PATH, SMALL_W2V_X_TEST_PATH, MEDIUM_DATA_PATH, MEDIUM_W2V_X_TRAIN_PATH, MEDIUM_W2V_X_TEST_PATH,ALL_CLASSES,\
    FULL_TFIDF_TEST_PATH, FULL_TFIDF_TRAIN_PATH
import keras
from keras.regularizers import l2
from sklearn.metrics import classification_report



def main():
    all_data = joblib.load(FULL_DATA_PATH)
    y_train = norm_names(all_data['y_train'])
    y_test = norm_names(all_data['y_test'])
    X_train = numpy.memmap(FULL_W2V_X_TRAIN_PATH, dtype=numpy.float32, mode='r', shape=(len(y_train), WORD2VEC_DIM))
    X_test = numpy.memmap(FULL_W2V_X_TEST_PATH,dtype=numpy.float32, mode='r', shape=(len(y_test), WORD2VEC_DIM))
    print(numpy.nanmax(X_train))
    prd = DAN()
    prd.train(X_train, X_test, y_train, y_test)


class DAN(object):
    BATCH_SIZE = 1

    def __init__(self):
        self.train_data = None

    def train(self, train_data, test_data, train_target, test_target):
        self.train_data = numpy.nan_to_num(train_data)
        self.test_data = numpy.nan_to_num(test_data)
        self.train_target = keras.utils.to_categorical(train_target, len(ALL_CLASSES))
        self.test_target = keras.utils.to_categorical(test_target, len(ALL_CLASSES))
        self.model = keras.models.Sequential()
        print(train_data, self.train_target)
        print(train_data[1700])
        callback = keras.callbacks.EarlyStopping(patience=10)
        self.model.add(keras.layers.Dense(300, input_shape=(self.train_data.shape[1],)))
        self.model.add(keras.layers.Dense(19, activation='softmax', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001)))
        opt = keras.optimizers.Adadelta(epsilon=1e-6)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.train_data, self.train_target, epochs=1000, validation_data=(self.test_data, self.test_target),callbacks=[callback], verbose=1, batch_size=16, shuffle=True)
        print(test_target,  list(self.model.predict_classes(self.test_data, verbose=0)))
        print(classification_report(test_target, list(self.model.predict_classes(self.test_data, verbose=0)), target_names=ALL_CLASSES))
        self.model.save('D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\trained_models\\nbow.hdf5')


if __name__ == '__main__':
    main()
