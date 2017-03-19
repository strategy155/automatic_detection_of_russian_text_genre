from gensim.models import Word2Vec
from utils import line_generator
import os
import joblib
import gensim
import nltk
import re
import multiprocessing as mp
import time


nltk.data.path.append('D:\\usr\\gwm\\materials\\nltk_data')
fn = 'word2vec_src.txt'
STOPS = nltk.corpus.stopwords.words('russian')


def worker(name, q):
    '''stupidly simulates long running process'''
    arg = joblib.load(name)
    word_list = re.sub('[^а-яА-яёЁ]', ' ', arg.lower()).split()
    filtered_words = [word for word in word_list  if word not in STOPS]
    res = ' '.join(filtered_words)
    q.put(res)
    return res

def listener(q):
    '''listens for messages on the q, writes to file. '''
    f = open(fn, 'wt')
    m=''
    c = 0
    while True:
        print(c, '377000')
        c = c + 1
        m = q.get()
        yield m




class MyCorp(object):
    def __iter__(self):
        manager = mp.Manager()
        q = manager.Queue()
        pool = mp.Pool(mp.cpu_count() + 2)
        # put listener to work first
        watcher = pool.apply_async(listener, (q,))
        # fire off workers
        jobs = []
        c=0
        rootd = os.walk('D:\\usr\\gwm\\materials\\c_w\\lib.rus.ec')
        with open('word2vec_src.txt', 'w', encoding='utf-8') as fopen:
            for root, dirs, filenames in rootd:
                for filename in filenames:
                    print(c, '377000')
                    c = c + 1
                    _abs_path = str(os.path.join(root, filename))
                    new_path = os.path.join(
                        os.path.join('D:\\usr\\gwm\\materials\\c_w\\full_strings', _abs_path.split('\\')[-2]),
                        (os.path.splitext(filename)[0] + '_line.pickle'))
                    job = pool.apply_async(worker, (new_path, q))
                    jobs.append(job)
        # collect results from the workers through the pool result queue
        c=0
        for job in jobs:
            job.get()
            print(c, '377000')
            c = c + 1

        # now we are done, kill the listener
        q.put('kill')
        pool.close()
if __name__ == '__main__':
    # main()
    corpus_friend = MyCorp()
    print(nltk.corpus.stopwords.words('russian'))
    print(Word2Vec(size=30000, window=10,workers=8,iter=5,sg=1).estimate_memory(3000000))
