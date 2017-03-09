import pickle
import joblib
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from concurrent.futures import ProcessPoolExecutor as Pool
from utils import get_text
import os


def work(root,filename):
    _abs_path = str(os.path.join(root, filename))
    new_path = os.path.join(os.path.join('D:\\usr\\gwm\\materials\\c_w\\full_strings', _abs_path.split('\\')[-2]),
                            (os.path.splitext(filename)[0] + '_line.pickle'))
    if not os.path.isfile(new_path):
        lines = get_text(_abs_path)
        print(_abs_path)
        big_line = ''
        for line in lines:
            big_line += line
        joblib.dump(big_line, new_path)
    return None








if __name__ == '__main__':
    rootd = os.walk('D:\\usr\\gwm\\materials\\c_w\\lib.rus.ec')
    for root, dirs, filenames in rootd:
        for filename in filenames:
            work(root,filename)