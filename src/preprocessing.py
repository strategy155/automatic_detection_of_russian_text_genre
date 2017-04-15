import re
from gensim.models.keyedvectors import KeyedVectors
import joblib
import numpy
import pymorphy2
from src.definitions import RUSVECTORES_PATH, ALL_DATA_PATH, SEQUENCE_LENGTH, WORD2VEC_DIM, ALL_CLASSES


morph = pymorphy2.MorphAnalyzer()


def convert_text_to_words_list(filename):
    _raw_string = joblib.load(filename)
    words = re.sub('[^а-яА-яёЁ]', ' ', _raw_string.lower()).split(' ')
    words = list(filter(None, words))
    return words


def create_list_of_w2v_vectors(words_list, model):
    new_words = []
    for word in words_list:
        try:
            new_words+=list(model.word_vec(_return_annotated(word)))
        except KeyError:
            continue
    return new_words


def _load_w2v_model(bin_path):
    model = KeyedVectors.load_word2vec_format(bin_path, binary=True)
    return model


def pad_sequences(sequences):
    new_array = numpy.zeros(SEQUENCE_LENGTH*WORD2VEC_DIM)
    for idx, elem in enumerate(sequences):
        new_array[idx] = numpy.asarray(elem)
        if idx == SEQUENCE_LENGTH*WORD2VEC_DIM:
            break
    return new_array


def get_w2v_dataset(X_filenames):
    full_length = len(X_filenames)

    for idx, filename in enumerate(X_filenames, start=1):
        _words_list = convert_text_to_words_list(filename)
        _model = _load_w2v_model(RUSVECTORES_PATH)
        _vectors = create_list_of_w2v_vectors(_words_list, _model)
        _padded_sequence = pad_sequences(_vectors)
        print(_padded_sequence.shape)
        if idx == 1:
            X = numpy.asarray(_padded_sequence)
        else:
            print(X.shape, _padded_sequence.shape)
            X = numpy.vstack((X, _padded_sequence))
        print(X.T.shape)
        if idx == 2:
            break
        print(idx, full_length)

    return X


def norm_names(y):
    new_y = []
    for elem in y:
        for idx, classname in enumerate(ALL_CLASSES):
            if classname == elem:
                new_y.append(idx)
                break
    return new_y

def create_w2v_marked_samples():
    filenames_by_genres = joblib.load(ALL_DATA_PATH)
    train_filenames = filenames_by_genres['X_train']
    test_filenames = filenames_by_genres['X_test']
    X_train = get_w2v_dataset(train_filenames)
    X_test = get_w2v_dataset(test_filenames)
    return X_train, X_test



def _return_annotated(word):
    _tag_vars = morph.parse(word)
    annotated = word + '_' + str(_tag_vars[0].tag.POS)
    return annotated


def main():
    print('che')

if __name__ == '__main__':
    main()