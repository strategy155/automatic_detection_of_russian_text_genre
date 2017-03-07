from sklearn.feature_extraction.text import TfidfVectorizer
from utils import line_generator



def main():
    tf_idf = TfidfVectorizer(input='content')
    tf_idf.fit(line_generator('D:\\usr\\gwm\\materials\\c_w\\lib.rus.ec'))
    # joblib.dump(tf_idf, 'tf_idf')

if __name__ == '__main__':
    main()
