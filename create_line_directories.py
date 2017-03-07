import pickle
import joblib
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from concurrent.futures import ProcessPoolExecutor as Pool
from utils import get_text
import os


def work(rootd,filename):
    _abs_path = str(os.path.join(rootd, filename))
    lines = get_text(_abs_path)
    big_line = ''
    for line in lines:
        big_line += line
    new_path = os.path.join(os.path.join('D:\\usr\\gwm\\materials\\c_w\\full_strings', _abs_path.split('\\')[-2]),
                            (os.path.splitext(filename)[0] + '_line.pickle'))
    joblib.dump(big_line, new_path)
    return None



def folder(rootd):
    for filename in rootd[2]:
        work(rootd[0], filename)
    return None





if __name__ == '__main__':
    with Pool() as pool:
        rootd = os.walk('D:\\usr\\gwm\\materials\\c_w\\lib.rus.ec')
        print(list(pool.map(folder, rootd)))