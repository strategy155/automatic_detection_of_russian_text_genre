import re
import pickle
import numpy
import os
import sys
from msvcrt import getch, kbhit


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
    for elem in soup.iter():
        try:
            if elem.tag == tag:
                text += str(elem.text)
        except ValueError:
            continue
    all_lines = list(filter(None,text.split('\n')))
    return all_lines


# def _genre_code_minimizer(genre_code):
#         if genre_code.startswith('det') or genre_code == 'thriller':
#             genre_code = 'det'
#         elif genre_code.startswith('sf'):
#             genre_code = 'sf'
#         elif genre_code.startswith('prose') or genre_code.startswith('proce') or genre_code.startswith('short'):
#             genre_code = 'prose'
#         elif genre_code.startswith('love') or genre_code.startswith('romance'):
#             genre_code = 'love'
#         elif genre_code.startswith('adv'):
#             genre_code = 'adv'
#         elif genre_code == 'dramaturgy':
#             genre_code = 'poetry'
#         elif genre_code.startswith('antique'):
#             genre_code = 'ant'
#         elif genre_code.startswith('sci'):
#             genre_code = 'sci'
#         elif genre_code.startswith('comp'):
#             genre_code = 'comp'
#         elif genre_code.startswith('ref'):
#             genre_code = 'ref'
#         elif genre_code.startswith('nonf') or genre_code == 'design':
#             genre_code = 'nonf'
#         elif genre_code.startswith('religion'):
#             genre_code = 'rel'
#         elif genre_code.startswith('humor'):
#             genre_code = 'hum'
#         elif genre_code.startswith('home'):
#             genre_code = 'home'
#         elif genre_code.startswith('child'):
#             genre_code = 'child'
#         else:
#             genre_code = 'other'
#         return genre_code


def _get_true_tag(soup):
    nsmap = soup.getroot().nsmap
    if nsmap[None] is None:
        tag = 'genre'
    else:
        tag = etree.QName(nsmap[None], 'genre')
    return tag


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
    signs_count = []
    for sign in signs:
        sign_count = 0
        for line in all_lines:
            sign_count += line.count(sign)
        signs_count.append(sign_count)
    return signs_count




def get_all_words(filename):
    soup = _get_soup(filename)
    all_lines = _get_text(soup)
    all_words = _get_all_words(all_lines)
    return all_words


def _array_of_features(filename):
    soup = _get_soup(filename)
    all_lines = _get_text(soup)
    signs = ['.', ',', ':', ';', '!', '?', '(', "'", '"', "`", '[', '-', '_', '/']
    signs_count = _count_all_signs(all_lines, signs)
    capital_count = _count_all_capitals(all_lines)
    words_count = _count_all_words(all_lines)
    signs_count.append(capital_count)
    array = numpy.array(signs_count)
    return numpy.divide(array,words_count)


def _count_genres(genre_list):
    all_len = 0
    c = 0
    for key in genre_list.keys():
        c=c+1
        try:
            all_len =  all_len + len(genre_list[key])
        except TypeError:
            continue
    print(all_len, all_len/(c-1))

    for key in genre_list.keys():
        if len(genre_list[key]) > 600:
            print(key)
    return None


if __name__ == '__main__':
    # with open('genre_list', 'rb') as f:
    #     data = pickle.load(f)
    #     _count_genres(data)
    c_max = 0
    flag = 0
    for rootd, dirs, filenames in os.walk('D:\\usr\\gwm\\materials\\c_w\\lib.rus.ec'):
        for filename in filenames:
            c_max = c_max + 1
    for rootd, dirs, filenames in os.walk('D:\\usr\\gwm\\materials\\c_w\\lib.rus.ec'):
        for filename in filenames:
            _abs_path = str(os.path.join(rootd,filename))
            with open(os.path.splitext(_abs_path)[0] + '_line.txt', 'wb') as f:
                pickle.dump(_array_of_features(_abs_path),f)
            if kbhit():
                if ord(getch()) == 27:  # ESC
                    sys.exit()
