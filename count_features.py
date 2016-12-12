import re
import numpy
import bs4
import os
import json
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer
import pickle
import io
import lxml.html
import html

try:
  from lxml import etree
  print("running with lxml.etree")
except ImportError:
  try:
    # Python 2.5
    import xml.etree.cElementTree as etree
    print("running with cElementTree on Python 2.5+")
  except ImportError:
    try:
      # Python 2.5
      import xml.etree.ElementTree as etree
      print("running with ElementTree on Python 2.5+")
    except ImportError:
      try:
        # normal cElementTree install
        import cElementTree as etree
        print("running with cElementTree")
      except ImportError:
        try:
          # normal ElementTree install
          import elementtree.ElementTree as etree
          print("running with ElementTree")
        except ImportError:
          print("Failed to import ElementTree from any known place")
parser = etree.XMLParser(ns_clean = True, recover = True, encoding='cp1251')


def _get_soup(filename):
    soup = etree.parse(filename, parser)
    return soup


def _get_text(soup):
    text = ''
    print(etree.tostring(soup).decode())
    print(soup)
    print(soup.findall('.//p'))
    all_lines = list(filter(None,text.split('\n')))
    print(all_lines)
    return all_lines


def _get_genre(soup):
    elem = soup.find('genre')
    try:
        if elem['match']:
            return 'match_genre'
    except (KeyError):
        return elem


def _count_all_words(all_lines):
    words_count = 0
    for line in all_lines:
        only_letters = re.sub('[^а-яА-яёЁ]', ' ', line)
        words = only_letters.split()
        words_count += len(words)
    return words_count


def _get_all_words(all_lines):
    all_words = []
    for line in all_lines:
        only_letters = re.sub('[^а-яА-яёЁ]', ' ', line)
        words = only_letters.lower().split()
        all_words+= words
    return all_words


def _count_all_capitals(all_lines):
    capital_count = 0
    for line in all_lines:
        words = line.split()
        for word in words:
            capital_count += int(word.isupper())
    return capital_count


def _count_all_signs(all_lines, signs):
    signs_count = {}
    for sign in signs:
        sign_count = 0
        for line in all_lines:
            sign_count += line.count(sign)
        signs_count[sign] = sign_count
    return signs_count


def _get_lines(filename):
    with open(filename, 'r', encoding='utf-8') as test_file:
        all_lines = test_file.readlines()
    return all_lines


def calibration_bag(filename, bag_of_words):
    soup = _get_soup(filename)
    all_lines = _get_text(soup)
    all_words = _get_all_words(all_lines)
    bag_of_words.fit(all_words)
    return bag_of_words

def _array_of_features(filename, bag_of_words):
    features_count = {}
    soup = _get_soup(filename)
    all_lines = _get_text(soup)
    signs = ['.', ',', ':', ';', '!', '?', '(', "'", '"', "`", '[', '-', '_', '/']
    signs_count = _count_all_signs(all_lines, signs)
    capital_count = _count_all_capitals(all_lines)
    words_count = _count_all_words(all_lines)
    all_words = _get_all_words(all_lines)
    bag_of_words.fit(all_words)
    print(bag_of_words)
    print(bag_of_words.vocabulary_)
    features_count['character'] = signs_count
    features_count['character']['capitals'] = capital_count
    features_count['words-count'] = words_count
    return features_count


def get_features_from_all_texts():
    class_names = []
    counter = 0
    genre_dict = {}
    num = 0
    for root, dirs, filenames in os.walk('D:\LibRu\_Lib.rus.ec - Официальная\lib.rus.ec'):
        for filename in filenames:
            num+=1
    # with open('genre_list_2.json','w') as file:
    #     for root, dirs, filenames in os.walk('D:\LibRu\_Lib.rus.ec - Официальная\lib.rus.ec'):
    #         for filename in filenames:
    #             counter+=1
    #             soup = _get_soup(os.path.join(root,filename))
    #             try:
    #                 elem_true_name = re.sub('<[^>]*>', '', _get_genre(soup).text)
    #             except AttributeError:
    #                 elem_true_name = 'no_genre'
    #             try:
    #                 genre_dict[elem_true_name].append(os.path.join(root,filename))
    #             except KeyError:
    #                 genre_dict[elem_true_name] =[os.path.join(root, filename)]
    #             print(num, counter)
    #     file.write(json.dumps(genre_dict, indent=4))
    features = dict()
    bag_of_words = TfidfVectorizer(input='content', analyzer='word')
    for root, dirs, filenames in os.walk('D:\LibRu\_Lib.rus.ec - Официальная\lib.rus.ec'):
        print(root)
        print(num)
        for filename in filenames:
            counter+=1
            print(filename)
            bag_of_words = calibration_bag(os.path.join(root, filename), bag_of_words)
            print(counter)
    print(bag_of_words.vocabulary_)
    with open('bag_dic', 'w') as dict_file:
        dict_file.write(json.dumps(bag_of_words.vocabulary_))
    with open('bag_vec.pk', 'wb') as vec_file:
        vec_file.write(pickle.dumps(bag_of_words))
    # print('init')
    # for elem in genre_filename.keys():
    #     counter+=1
    #     elem_true_name = re.sub('<[^>]*>','',elem)
    #     features[elem_true_name] = []
    #     for root, dirs, filenames in os.walk('D:\LibRu\_Lib.rus.ec - Официальная\lib.rus.ec'):
    #         for filename in filenames:
    #             if filename in genre_filename[elem]:
    #                 bag_of_words = CountVectorizer(input='content', analyzer='word')
    #                 try:
    #                     features[elem_true_name].append(_array_of_features(os.path.join(root,filename), bag_of_words))
    #                     class_names.append(elem_true_name)
    #                 except ZeroDivisionError:
    #                     continue
    #     print(len(list(genre_filename.keys())), counter)
    # print(features)
    # print(class_names)
    # with(open('genre_data_3.json', 'a')) as file_output:
    #     file_output.write(json.dumps(features, ensure_ascii=False, indent=4))
get_features_from_all_texts()