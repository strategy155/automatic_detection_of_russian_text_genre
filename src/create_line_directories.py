import os

import joblib


def work(root,filename):
    _abs_path = str(os.path.join(root, filename))

    l = joblib.load(_abs_path)
    print(l)
    return None








if __name__ == '__main__':
    rootd = os.walk('D:\\usr\\gwm\\materials\\c_w\\full_strings')
    c=0
    for root, dirs, filenames in rootd:
        for filename in filenames:
            c+=1
            print(c)
            work(root,filename)