import re
import numpy


def count_all_words(all_lines):
    words_count = 0
    for line in all_lines:
        only_letters = re.sub('[^а-яА-яёЁ]', ' ', line)
        words = only_letters.split()
        words_count += len(words)
    return words_count

def count_all_capitals(all_lines):
    capital_count = 0
    for line in all_lines:
        words = line.split()
        for word in words:
            capital_count += int(word.isupper())
    return capital_count

def count_all_signs(all_lines, signs):
    signs_count = []
    for sign in signs:
        sign_count = 0
        for line in all_lines:
            sign_count += line.count(sign)
        signs_count.append(sign_count)
    return signs_count


def get_lines(filename):
    with open(filename, 'r', encoding='utf-8') as test_file:
        all_lines = test_file.readlines()
    return all_lines


def make_features_array(words_count, features_count):
    features_array=numpy.array(features_count)
    correction_coef = 1/words_count
    features_transformed = numpy.dot(features_array, correction_coef)
    return features_transformed

def main():
    all_lines = get_lines('sample.txt')
    signs = ['.', ',', ':', ';', '!', '?', '(', '"', "`", '[', '-', '_', '/']
    signs_count = count_all_signs(all_lines, signs)
    capital_count = count_all_capitals(all_lines)
    words_count = count_all_words(all_lines)
    features_count = signs_count
    features_count.append(capital_count)
    print(make_features_array(words_count, features_count))


if __name__ == '__main__':
    main()