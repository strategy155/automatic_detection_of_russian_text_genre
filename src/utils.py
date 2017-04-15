import os
import re
import sys
from lxml import etree
from msvcrt import getch, kbhit


def get_soup(filename):
    parser = etree.XMLParser(ns_clean=True, recover=True)
    soup = etree.parse(filename, parser)
    return soup


def check_if_key_pressed(ord_code):
    if kbhit():
        if ord(getch()) == ord_code:
            return True


def get_all_tags(soup, tag):
    tags = []
    for elem in soup.iter():
        if elem.tag == tag:
            try:
                tags.append(str(elem.text))
            except AttributeError:
                tags = ['no_tag']
    if tags == []:
        tags = ['no_tag']
    return tags



def get_true_tag(soup, base_tag):
    nsmap = soup.getroot().nsmap
    if nsmap[None] is None:
        tag = base_tag
    else:
        try:
            tag = etree.QName(nsmap[None], base_tag)
        except ValueError:
            tag = 'no_tag'
    return tag


def get_text(filename):
    soup = get_soup(filename)
    text = ''
    nsmap = soup.getroot().nsmap
    try:
        tag = etree.QName(nsmap[None],'p')
    except ValueError:
        tag = 'p'
    except KeyError:
        tag = 'p'
    for elem in soup.iter():
        try:
            if elem.tag == tag:
                text += ' ' +str(elem.text)
        except ValueError:
            continue
    all_lines = list(filter(None,text.split('\n')))
    return all_lines


def get_all_words(filename):
    all_lines = get_text(filename)
    all_words = []
    for line in all_lines:
        only_letters = re.sub('[^а-яА-яёЁ]', ' ', line)
        words = only_letters.lower().split()
        all_words += words
    return all_words


def line_generator(root):
    for rootd, dirs, filenames in os.walk(root):
        for filename in filenames:
            _abs_path = str(os.path.join(rootd,filename))
            lines = get_text(_abs_path)
            check_if_key_pressed(27)
            for line in lines:
                yield line




def check_lang(filename, lang='ru'):
    soup = get_soup(filename)
    lang_tag = get_true_tag(soup, 'lang')
    langs = get_all_tags(soup, lang_tag)
    if langs == ['ru']:
        return True
    else:
        return False