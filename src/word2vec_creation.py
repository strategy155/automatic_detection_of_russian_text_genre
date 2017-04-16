import os
import re
import gensim
import joblib
import nltk
import pickle
from datetime import datetime, date
from src.utils import check_if_key_pressed
import time


nltk.data.path.append('D:\\usr\\gwm\\materials\\nltk_data')
STOPS = nltk.corpus.stopwords.words('russian')


class Word2Vec_Corp(object):
    def __init__(self):
        self.flag = 0
        self.epoch = 0

    def __iter__(self):
        rootd = os.walk('D:\\usr\\gwm\\materials\\c_w\\full_strings')
        c = 0
        c_max = 0
        self.flag = 0
        try:
            with open('tmp\\technical_information.pickle','rb') as tech_inf:
                inf = pickle.load(tech_inf)
                self.epoch = inf[1]
                c_max = inf[0]
        except FileNotFoundError:
            self.epoch = 0
            pass
        for root, dirs, filenames in rootd:
            for filename in filenames:
                if self.flag == 1:
                    break
                if c < c_max:
                    c+=1
                    continue
                if check_if_key_pressed(27):
                    with open('tmp\\technical_information.pickle','wb') as fin:
                        pickle.dump([c, self.epoch],fin)
                    self.flag = 1
                    break
                c += 1
                print(c)
                print(self.epoch, self.flag)
                _abs_path = str(os.path.join(root, filename))
                arg = joblib.load(_abs_path)
                word_seq = re.sub('[^а-яА-яёЁ]', ' ', arg.lower()).split()
                word_seq = [x for x in word_seq if x not in STOPS]
                yield word_seq
        if self.flag != 1:
            self.epoch += 1
            with open('tmp\\technical_information.pickle', 'wb') as fin:
                pickle.dump([0, self.epoch], fin)


def main():
    corp = MyCorp()
    try:
        model = gensim.models.Word2Vec.load('w2v')
        while corp.epoch != 5 and corp.flag != 1:
            corp.epoch+=1
            model.train(corp)
    except FileNotFoundError:
        pass
        model = gensim.models.Word2Vec(sentences=corp,size=1000,workers=8,window=10,iter=5,batch_words=100000)
    model.save('w2v')



if __name__ == '__main__':
    start = datetime.today()
    time.sleep(10)
    print(datetime.today()-start)