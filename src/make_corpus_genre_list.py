import os
import pickle

from src.utils import get_soup, get_true_tag, get_all_tags, check_lang

CORPUS_PATH = 'D:\\usr\\gwm\\materials\\c_w\\lib.rus.ec'


def _modify_genre_list(filename, genre_list):
    soup = get_soup(filename)
    genre_tag = get_true_tag(soup, 'genre')
    genres = get_all_tags(soup, genre_tag)
    if check_lang(filename, 'ru'):
        for genre in genres:
            try:
                genre_list[genre].append(filename)
            except KeyError:
                genre_list[genre] = [filename]
    return None


def _get_genre_list(root):
    genre_list = {}
    for rootd, dirs, filenames in os.walk(root):
        for filename in filenames:
            _abs_path = str(os.path.join(rootd,filename))
            _modify_genre_list(_abs_path, genre_list)
    return genre_list


def _save_file(filename,data):
    _file = open(filename, 'wb')
    pickle.dump(data, _file)
    return None

def main():
    _genre_list = _get_genre_list(CORPUS_PATH)
    _save_file('genre_list', _genre_list)
    return None


if __name__ == '__main__':
    main()
