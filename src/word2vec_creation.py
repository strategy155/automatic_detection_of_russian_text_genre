import os
import re

import gensim
import joblib
import nltk

nltk.data.path.append('D:\\usr\\gwm\\materials\\nltk_data')
fn = 'word2vec_src.txt'
STOPS = nltk.corpus.stopwords.words('russian')


class MyCorp(object):
    def __iter__(self):
        rootd = os.walk('D:\\usr\\gwm\\materials\\c_w\\full_strings')
        c = 0
        for root, dirs, filenames in rootd:
            for filename in filenames:
                c+=1
                print(c)
                _abs_path = str(os.path.join(root, filename))
                arg = joblib.load(_abs_path)
                word_seq = re.sub('[^а-яА-яёЁ]', ' ', arg.lower()).split()
                word_seq = [x for x in word_seq if x not in STOPS]
                yield word_seq
                # collect results from the workers through the pool result queue


if __name__ == '__main__':
    # main()
    corp = MyCorp()
    model = gensim.models.Word2Vec(sentences=corp,size=1000,workers=8,window=10,iter=5,batch_words=100000)
    model.save('w2v')