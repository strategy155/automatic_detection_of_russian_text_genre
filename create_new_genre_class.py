import joblib
import os


def _create_good_dic(max_number = 50):
    data = joblib.load('genre_list')
    data.pop('tech', None)
    new_data = {}
    abs_len = 0
    sup_len=0
    c = 0
    for key in data.keys():
        abs_len+= len(data[key])
        if len(data[key])> 100:
            sup_len += len(data[key])
            _check_genre('sf',data,new_data,key)
            _check_genre('det', data, new_data,key)
            _check_genre('prose', data, new_data,key)
            _check_genre('love', data, new_data,key)
            _check_genre('adv', data, new_data,key)
            _check_genre('child', data, new_data,key)
            _check_genre('antique', data, new_data,key)
            _check_genre('sci', data, new_data,key)
            _check_genre('comp', data, new_data,key)
            _check_genre('ref', data, new_data,key)
            _check_genre('nonf', data, new_data,key)
            _check_genre('religi', data, new_data,key)
            _check_genre('humor', data, new_data,key)
            _check_genre('home', data, new_data,key)
            print(key)
    new_len = 0
    for key in new_data.keys():
        new_len += len(new_data[key])
    print(abs_len, sup_len,new_len, new_data)
    joblib.dump(new_data,"new_genre_list")
    return None


def _check_genre(name,old_dic,new_dic,key):
    if key.startswith(name):
        try:
            assert new_dic[name]
        except KeyError:
            new_dic[name]=[]
        for elem in old_dic[key]:
            new_path = os.path.join(os.path.join('D:\\usr\\gwm\\materials\\c_w\\full_strings', elem.split('\\')[-2]),
            (os.path.splitext(elem.split('\\')[-1])[0] + '_line.pickle'))
            new_dic[name].append(new_path)
    return None


if __name__ == '__main__':
    _create_good_dic()