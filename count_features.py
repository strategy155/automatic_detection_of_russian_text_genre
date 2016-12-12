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
parser = etree.XMLParser(ns_clean = True, recover = True)


def _get_soup(filename):
    soup = etree.parse(filename, parser)
    return soup


def _get_text(soup):
    text = ''
    nsmap = soup.getroot().nsmap
    tag = etree.QName(nsmap[None],'p')
    print(tag)
    for elem in soup.iter():
        if elem.tag == tag:
            text += str(elem.text)
    all_lines = list(filter(None,text.split('\n')))
    print(all_lines)
    return all_lines


def _get_genre(soup):
    genres = {}
    nsmap = soup.getroot().nsmap
    tag = etree.QName(nsmap[None],'genre')
    for elem in soup.iter():
        if elem.tag == tag:
            genre_code = str(elem.text)
            if genre_code.startswith('det') or genre_code == 'thriller':
                genre_code = 'det'
            elif genre_code.startswith('sf'):
                genre_code = 'sf'
            elif genre_code.startswith('prose') or genre_code.startswith('proce') or genre_code.startswith('short'):
                genre_code = 'prose'
            elif genre_code.startswith('love') or genre_code.startswith('romance'):
                genre_code = 'love'
            elif genre_code.startswith('adv'):
                genre_code = 'adv'
            elif genre_code == 'dramaturgy':
                genre_code = 'poetry'
            elif genre_code.startswith('antique'):
                genre_code = 'ant'
            elif genre_code.startswith('sci'):
                genre_code = 'sci'
            elif genre_code.startswith('comp'):
                genre_code = 'comp'
            elif genre_code.startswith('ref'):
                genre_code = 'ref'
            elif genre_code.startswith('nonf') or genre_code == 'design':
                genre_code = 'nonf'
            elif genre_code.startswith('religion'):
                genre_code = 'rel'
            elif genre_code.startswith('humor'):
                genre_code = 'hum'
            elif genre_code.startswith('home'):
                genre_code = 'home'
            elif genre_code.startswith('child'):
                genre_code = 'child'
            else:
                genre_code = 'other'
            try:
                genres[genre_code]+=1
            except KeyError:
                genres[genre_code] = 1
    c = 0
    c_k = ''
    for key in genres.keys():
        if genres[key]> c:
            c = genres[key]
            c_k = key
    return c_k


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
        all_words += words
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


def calibration_bag(filename, hehe):
    soup = _get_soup(filename)
    all_lines = _get_text(soup)
    all_words = _get_all_words(all_lines)
    return hehe.fit_transform(all_words)


def _array_of_features(filename, bag_of_words):
    features_count = {}
    soup = _get_soup(filename)
    all_lines = _get_text(soup)
    signs = ['.', ',', ':', ';', '!', '?', '(', "'", '"', "`", '[', '-', '_', '/']
    signs_count = _count_all_signs(all_lines, signs)
    capital_count = _count_all_capitals(all_lines)
    words_count = _count_all_words(all_lines)
    all_words = _get_all_words(all_lines)
    bag_of_words.transform(all_words)
    features_count['character'] = signs_count
    features_count['character']['capitals'] = capital_count
    features_count['words-count'] = words_count
    return features_count


def get_features_from_all_texts():
    # class_names = []
    # counter = 0
    # genre_dict = {}
    # num = 0
    # for root, dirs, filenames in os.walk('D:\LibRu\_Lib.rus.ec - Официальная\lib.rus.ec'):
    #     for filename in filenames:
    #         num+=1
    # with open('genre_list_2.json','w') as file:
    #     for root, dirs, filenames in os.walk('D:\LibRu\_Lib.rus.ec - Официальная\lib.rus.ec'):
    #         for filename in filenames:
    #             counter+=1
    #             soup = _get_soup(os.path.join(root,filename))
    #             for genre in genres:
    #                 genre_dict[genre].append(os.path.join(root,filename))
    #             print(num, counter)
    #     file.write(json.dumps(genre_dict, indent=4))
    soup = _get_soup('sample.fb2')
    print(_get_genre(soup))
    bag_of_words = TfidfVectorizer(input='content', analyzer='word')
    # for root, dirs, filenames in os.walk('D:\LibRu\_Lib.rus.ec - Официальная\lib.rus.ec'):
    #     print(root)
    #     print(num)
    #     for filename in filenames:
    #         counter+=1
    #         print(filename)
    #         bag_of_words = calibration_bag(os.path.join(root, filename), bag_of_words)
    #         print(counter)
    # with open('bag_vec.pk', 'wb') as vec_file:
    #     vec_file.write(pickle.dumps(bag_of_words))
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