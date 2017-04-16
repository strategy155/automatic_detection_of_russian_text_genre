import re
from gensim.models.keyedvectors import KeyedVectors
import numpy
import pymorphy2
import joblib
import os
from src.definitions import RUSVECTORES_PATH, SMALL_DATA_PATH, SEQUENCE_LENGTH, WORD2VEC_DIM, ALL_CLASSES,\
    ANNOTATED_CORPUS_PATH, FULL_DATA_PATH, MEDIUM_W2V_X_TEST_PATH, MEDIUM_W2V_X_TRAIN_PATH, SMALL_W2V_X_TRAIN_PATH, SMALL_W2V_X_TEST_PATH, MEDIUM_DATA_PATH, STOPS


morph = pymorphy2.MorphAnalyzer()


def convert_text_to_words_list(filename):
    _raw_string = joblib.load(filename)
    words = re.sub('[^а-яА-яёЁ]', ' ', _raw_string.lower()).split(' ')
    words = list(filter(None, words))
    words = [word for word in words if word not in STOPS]
    return words


def create_list_of_annotated(words_list):
    new_words = []
    for word in words_list:
        try:
            new_words.append(_return_annotated(word))
        except KeyError:
            continue
    return new_words


def dump_annotated_file(filename, annotated_list):
    new_path = return_annotated_path(filename)
    joblib.dump(annotated_list,new_path)
    return None


def return_annotated_path(filename):
    new_filename = str(os.path.splitext(filename.split('\\')[-1])[0].split('_')[0]) + '.pkl'
    new_path = os.path.join(ANNOTATED_CORPUS_PATH, new_filename)
    return new_path


def dump_annotated_corpus(X_filenames):
    for idx, filename in enumerate(X_filenames, start=1):
        _words_list = convert_text_to_words_list(filename)
        _annotated_text = create_list_of_annotated(_words_list)
        dump_annotated_file(filename, _annotated_text)
        print(idx)


def _load_w2v_model(bin_path):
    model = KeyedVectors.load_word2vec_format(bin_path, binary=True)
    return model


def pad_sequences(sequences):
    new_array = numpy.zeros(SEQUENCE_LENGTH*WORD2VEC_DIM)
    for idx, elem in enumerate(sequences):
        if idx == SEQUENCE_LENGTH*WORD2VEC_DIM:
            break
        new_array[idx] = numpy.asarray(elem)
    return new_array


def get_list_of_tags(model):
    inf_set = set()
    for elem in model.vocab.keys():
        inf_set.add(str(elem).split('_')[1])
    return list(inf_set)


def dump_w2v_dataset(X_filenames, path):
    X = numpy.memmap(path, mode='w+', dtype='float64', shape=(len(X_filenames),SEQUENCE_LENGTH*WORD2VEC_DIM))
    for idx, filename in enumerate(X_filenames, start=0):
        _model = _load_w2v_model(RUSVECTORES_PATH)
        _new_path = return_annotated_path(filename)
        _annotated_word_list = joblib.load(_new_path)
        _vectors = create_list_of_w2v_vectors(_annotated_word_list, _model)
        _padded_sequence = pad_sequences(_vectors)
        X[idx] = _padded_sequence
        print(idx, _padded_sequence.shape, X.shape, X[idx], len(_annotated_word_list))
    return None


def create_list_of_w2v_vectors(annotated_word_list, model):
    vecs = []
    tag_set = get_list_of_tags(model)
    for elem in annotated_word_list:
        try:
            vec = model.word_vec(elem)
            vecs+= list(vec)
        except KeyError:
            for tag in tag_set:
                try:
                    new_annotated = str(elem).split('_')[0] + '_' + tag
                    vecs += list(model.word_vec(new_annotated))
                    break
                except KeyError:
                    continue
            continue
    return vecs


def norm_names(y):
    new_y = []
    for elem in y:
        for idx, classname in enumerate(ALL_CLASSES):
            if classname == elem:
                new_y.append(idx)
                break
    return new_y

def create_annotated_corpus():
    filenames_by_genres = joblib.load(SMALL_DATA_PATH)
    train_filenames = filenames_by_genres['X_train']
    test_filenames = filenames_by_genres['X_test']
    dump_annotated_corpus(train_filenames)
    dump_annotated_corpus(test_filenames)
    return None


def map_ud_tags_to_pymorphy(tag):
    mapping_dict={
        'ADJF': 'ADJ',
        'ADJS': 'ADJ',
        'COMP': 'ADV',
        'INFN': 'VERB',
        'PRTF': 'VERB',
        'PRTS': 'VERB',
        'NUMR': 'NUM',
        'CONJ': 'CCONJ',
        'PRED': 'ADV',
        'PRCL': 'PART',
        'PREP': 'ADP',
        'ADVB': 'ADV',
        'GRND': 'ADJ',
        'NPRO': 'PRON'
    }
    try:
        return mapping_dict[tag]
    except KeyError:
        return tag

def _return_annotated(word):
    _tag_vars = morph.parse(word)
    annotated = _tag_vars[0].normal_form + '_' + map_ud_tags_to_pymorphy(str(_tag_vars[0].tag.POS))
    return annotated


def main():
    filenames_by_genres = joblib.load(SMALL_DATA_PATH)
    train_filenames = filenames_by_genres['X_train']
    test_filenames = filenames_by_genres['X_test']
    dump_w2v_dataset(train_filenames, SMALL_W2V_X_TRAIN_PATH)
    dump_w2v_dataset(test_filenames, SMALL_W2V_X_TEST_PATH)
    # dump_annotated_corpus(train_filenames)
    # dump_annotated_corpus(test_filenames)


if __name__ == '__main__':
    main()