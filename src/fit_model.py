import joblib
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from src.preprocessing import create_w2v_marked_samples, norm_names
from src.definitions import ALL_DATA_PATH, SEQUENCE_LENGTH, WORD2VEC_DIM
from sklearn.metrics import classification_report

def main():
    all_data = joblib.load(ALL_DATA_PATH)
    X_train, X_test = create_w2v_marked_samples()
    y_train = to_categorical(norm_names(all_data['y_train']))
    y_test = to_categorical(norm_names(all_data['y_test']))
    NBOW(X_train,X_test,y_train,y_test)


def NBOW(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(38, input_shape=(SEQUENCE_LENGTH*WORD2VEC_DIM,), activation='sigmoid'))
    model.add(Dense(19, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))



if __name__ == '__main__':
    main()
