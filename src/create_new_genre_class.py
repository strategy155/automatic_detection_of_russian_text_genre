import os

import joblib


def _create_good_dic(max_number = 50):
    data = joblib.load('genre_list')
    data.pop('tech', None)
    new_data = {}
    abs_len = 0
    sup_len=0
    c = 0
    for k in sorted(data, key=lambda k: len(data[k]), reverse=True):
        print(k, len(data[k]))

    for key in data.keys():
        # x = 0
        # abs_len+= len(data[key])
        if len(data[key])>= 110:
            new_data[key] = []
            for elem in data[key]:
                new_path = os.path.join(
                    os.path.join('D:\\usr\\gwm\\materials\\c_w\\full_strings', elem.split('\\')[-2]),
                    (os.path.splitext(elem.split('\\')[-1])[0] + '_line.pickle'))
                new_data[key].append(new_path)

            # sup_len += len(data[key])
            # for elem in ALL_CLASSES:
            #     x += _check_genre(elem, data, new_data, key)

    # new_len = 0
    # for key in new_data.keys():
    #     new_len += len(new_data[key])
    joblib.dump(new_data,"test_lul")
    print(abs_len)
    return None


def _check_genre(name,old_dic,new_dic,key):
    if name in key:
        try:
            assert new_dic[name]
        except KeyError:
            new_dic[name]=[]
        for elem in old_dic[key]:
            new_path = os.path.join(os.path.join('D:\\usr\\gwm\\materials\\c_w\\full_strings', elem.split('\\')[-2]),
            (os.path.splitext(elem.split('\\')[-1])[0] + '_line.pickle'))
            new_dic[name].append(new_path)
        return 1
    return 0


if __name__ == '__main__':
    _create_good_dic()