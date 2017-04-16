import joblib
import numpy
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from src.preprocessing import norm_names
from src.definitions import SMALL_DATA_PATH, SEQUENCE_LENGTH, WORD2VEC_DIM, SMALL_W2V_X_TRAIN_PATH, SMALL_W2V_X_TEST_PATH
from sklearn.metrics import classification_report

def main():
    all_data = joblib.load(SMALL_DATA_PATH)
    y_train = to_categorical(norm_names(all_data['y_train']))
    y_test = to_categorical(norm_names(all_data['y_test']))
    X_train= numpy.memmap(SMALL_W2V_X_TRAIN_PATH, mode='r', shape=(len(y_train),SEQUENCE_LENGTH*WORD2VEC_DIM))
    X_test = numpy.memmap(SMALL_W2V_X_TEST_PATH, mode='r', shape=(len(y_test),SEQUENCE_LENGTH*WORD2VEC_DIM))
    print(X_train.shape)
    NBOW(X_train,X_test,y_train,y_test)


def NBOW(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(64, input_shape=(SEQUENCE_LENGTH*WORD2VEC_DIM,)))
    model.add(Dense(19, activation='softmax'))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy')
    model.fit(X_train, y_train, epochs=1000)
    y_pred = model.predict_classes(X_test)
    print(y_pred)
    print(classification_report(y_test,y_pred))


if __name__ == '__main__':
    main()
