import nltk


ANNOTATED_CORPUS_PATH = 'D:\\usr\\gwm\\materials\\c_w\\annotated'
CORPUS_PATH = 'D:\\usr\\gwm\\c_w\\lib.rus.ec'
ALL_CLASSES = ['sf', 'det', 'prose', 'love', 'adv', 'child', 'antique', 'sci', 'comp', 'ref', 'nonf', 'religi', 'humor', 'home', 'poetry', 'dramaturgy', 'periodic', 'other', 'russian_cont']
MAX_NUMBER = 100
PREDICTION_DATA = {'X_train': [], 'y_train': [], 'X_test': [], 'y_test': []}
X_TRAIN_PATH = 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\train_samples\\w2v_train'
SMALL_DATA_PATH = 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\train_data\\small_balanced_dataset_dictionary.pkl'
MEDIUM_DATA_PATH = 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\train_data\\medium_balanced_dataset_dictionary.pkl'
FULL_DATA_PATH = 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\train_data\\full_balanced_dataset_dictionary.pkl'
X_TEST_PATH = 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\train_samples\\w2v_test'
SMALL_W2V_X_TEST_PATH = 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\train_samples\\small_w2v_test.bin'
SMALL_W2V_X_TRAIN_PATH = 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\train_samples\\small_w2v_train.bin'
MEDIUM_W2V_X_TEST_PATH = 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\train_samples\\medium_w2v_test.bin'
MEDIUM_W2V_X_TRAIN_PATH = 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\train_samples\\medium_w2v_train.bin'
FULL_W2V_X_TEST_PATH = 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\train_samples\\full_w2v_test.bin'
FULL_W2V_X_TRAIN_PATH = 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\train_samples\\full_w2v_train.bin'
FULL_TFIDF_TRAIN_PATH = 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\train_samples\\full_tf-idf_format_prepared_train.pkl'
FULL_TFIDF_TEST_PATH =  'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\train_samples\\full_tf-idf_format_prepared_test.pkl'
RUSVECTORES_PATH = 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\models\\ruwikiruscorpora_0_300_20.bin'
SEQUENCE_LENGTH = 2000
WORD2VEC_DIM = 300
nltk.data.path.append('D:\\usr\\gwm\\materials\\nltk_data')
STOPS = nltk.corpus.stopwords.words('russian')