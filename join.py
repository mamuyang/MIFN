import html
import re
import nltk
from nltk import word_tokenize, pos_tag
from util import *

wnl = nltk.stem.WordNetLemmatizer()
stemer = nltk.stem.snowball.EnglishStemmer()


def lemmatize_and_stem_all(sentence):
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith('NN'):
            yield stemer.stem(wnl.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            yield stemer.stem(wnl.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
            yield stemer.stem(wnl.lemmatize(word, pos='a'))
        elif tag.startswith('R'):
            yield stemer.stem(wnl.lemmatize(word, pos='r'))
        else:
            yield stemer.stem(word)


fname = 'film_book.txt'
mte = 'movie_title_entity.txt'
bte = 'book_title_entity.txt'

encoding = 'utf-8'

# def process1():
#     with open('data/' + mte, 'r', encoding=encoding) as f1:
#         with open('data/disposed/' + mte, 'w+', encoding=encoding) as f2:
#             for line in f1:
#                 title = line.strip().split('\t')[1]
#                 title = process(title, 0)
#                 f2.write(title + '\n')
#
#
# def process2():
#     with open('data/' + bte, 'r', encoding=encoding) as f1:
#         with open('data/disposed/' + bte, 'w+', encoding=encoding) as f2:
#             for line in f1:
#                 title = line.strip().split('\t')[1]
#                 title = process(title, 1)
#                 f2.write(title + '\n')
#
#
# def sort_title(flag):
#     entity = mte if flag == 0 else bte
#     with open('data/disposed/' + entity, 'r', encoding=encoding) as f:
#         with open('data/disposed/sorted/' + entity, 'w+', encoding=encoding) as w:
#             list = [process(line.strip(), flag=flag) for line in f]
#             list.sort()
#             for l in list:
#                 w.write(l + '\n')
#
#
# def wash(flag):
#     entity = mte if flag == 0 else bte
#     with open('data/disposed/' + entity, 'w+', encoding=encoding) as f0:
#         with open('data/' + entity, 'r', encoding=encoding) as f1:
#             for line in f1:
#                 pair = line.strip().split('\t')
#                 f0.write(pair[0] + '\t' + trans(pair[1]) + '\n')
"""
check correctness of methods in util for preprocessing.
"""
# wash(0)
# wash(1)
# process1()
# process2()
# sort_title(0)
# sort_title(1)

def process_entity():
    root = 'data/'
    i, j = 0, 0
    with open(root + 'p_' + mte, 'w+', encoding=encoding) as f0:
        with open(root + mte, 'r', encoding=encoding) as f:
            for title in f:
                f0.write(process(title.strip().split('\t')[1], 0) + '\n')
                i += 1
                print(i)
    with open(root + 'p_' + bte, 'w+', encoding=encoding) as f0:
        with open(root + bte, 'r', encoding=encoding) as f:
            for title in f:
                f0.write(process(title.strip().split('\t')[1], 0) + '\n')
                j += 1
                print(j)


pattern = re.compile('\w')


def match(title1, title2, all=False):
    if title1 == title2:
        return True
    len1, len2 = len(title1), len(title2)
    if len1 <= len2:
        word_num = title1.count(' ') + 1
        if word_num > 1 or 1 <= (title2.count(' ') + 1) / word_num <= 2:
            result = re.search(re.escape(title1), title2, re.I)
            if result:
                span = result.span()
                left, right = True, True
                if span[0] > 0:
                    if re.match(pattern, title2[span[0] - 1]):
                        left = False
                if span[1] < len2:
                    if re.match(pattern, title2[span[1]]):
                        right = False
                if left and right:
                    return True
    elif all:
        word_num = title2.count(' ')+1
        if word_num > 2 or 1 <= (title1.count(' ') + 1) / (title2.count(' ') + 1) <= 2:
            result = re.search(re.escape(title2), title1, re.I)
            if result:
                span = result.span()
                left, right = True, True
                if span[0] > 0:
                    if re.match(pattern, title1[span[0]-1]):
                        left = False
                if span[1] < len1:
                    if re.match(pattern, title1[span[1]]):
                        right = False
                if left and right:
                    return True
    return False

def ceiling(ptmp, real, tmp, entity):
    bcount = ptmp.count(' ')
    if ptmp.count(' ') <= 3 and len(tmp) >= 15 + 5 * bcount:
        t1 = []
        for t in tmp:
            res = re.search(re.escape(ptmp), entity[t], flags=re.I)
            if res:
                if re.match(' [^A-Za-z,\']| *$', entity[t][res.span()[1]:]):
                    if res.span()[0] == 0:
                        t1.append(t)
                    elif res.span()[0] >= 2 \
                            and re.match('[^\w,\']{2}', entity[t][res.span()[0]-2:res.span()[0]]):
                        t1.append(t)
        tmp = t1
        # if len(tmp) >= 20 + 5 * bcount:
        #     t1 = []
        #     for t in tmp:
        #         res = re.search(re.escape(ptmp), entity[t], flags=re.I)
        #         if res:
        #             if res.span()[0] == 0:
        #                 if re.match(' *[:\-;&(\[]| *$', entity[t][res.span()[1]:]):
        #                     t1.append(t)
        #             elif entity[t][res.span()[0]-1] == '(' \
        #                     and re.match(' ?(,[\w.\-]+ ?\d+)?\)', entity[t][res.span()[1]:]):
        #                 t1.append(t)
        #     tmp = t1
    return tmp


def consider_hyphen(title):
    result = [title]
    if re.search('([^ ])-([^ ])', title):
        result.append(re.sub('([^ ])-([^ ])', r'\1\2', title))
        result.append(re.sub('([^ ])-([^ ])', r'\1 \2', title))
    return result


def exist_match(ptmp, tmp):
    if match(ptmp, tmp):
        return True
    list1 = consider_hyphen(ptmp)
    list2 = consider_hyphen(tmp)
    if len(list1) > 1 and len(list2) > 1:
        for x in consider_hyphen(ptmp):
            for y in consider_hyphen(tmp):
                if x == ptmp and y == tmp:
                    continue
                if match(x, y):
                    return True
    return False


def join(line, len_m, film_list, film_list_real, len_b, book_list, book_list_real, count, num, f_left, f_join):
    pair = line.strip().split('\t')
    tmp1 = []
    tmp2 = []
    ptmp = trans_sensitive(cut(pair[0]))
    for mi in range(len_m):
        # if pair[0] in movie or movie in pair[0]:
        tmp = film_list[mi]
        if tmp == '':
            continue
        if exist_match(ptmp, tmp):
            tmp1.append(mi)
            # print([pair[0], tmp])
    tmp1 = ceiling(ptmp, pair[0], tmp1, film_list)
    if tmp1:
        ptmp = cut(pair[1])
        for bi in range(len_b):
            # if pair[1] in book or book in pair[1]:
            tmp = book_list[bi]
            if tmp == '':
                continue
            if exist_match(ptmp, tmp):
                tmp2.append(bi)
                # print(tmp)
            # print([pair[1], tmp])

        tmp2 = ceiling(ptmp, pair[1], tmp2, book_list)
        if tmp2:
            num += len(tmp1) * len(tmp2)
            count += 1
            print(pair, len(tmp1), len(tmp2))
            # f0.write('\t'.join(pair))
            # f0.write('\t' + str(len(tmp1)) + '\t' + str(len(tmp2)) + '\n')

            f_left.write(pair[1] + '\t' + pair[0] + '\n')
            for t1 in tmp1:
                for t2 in tmp2:
                    f_join.write(book_list_real[t2] + '\t' + film_list_real[t1] + '\n')
    return num, count



def full_join():
    root = 'data/'
    film_list = []
    film_list_real = []
    book_list = []
    book_list_real = []
    with open(root + 'film_book_left.txt', 'w+', encoding=encoding) as f_left:
        with open(root + 'full_join.txt', 'w+', encoding=encoding) as f_join:
            with open(root + mte, 'r', encoding=encoding) as f:
                for title in f:
                    film_list_real.append(title.strip())
            # with open(root + 'p_' + mte, 'r', encoding=encoding) as f:
            #     for title in f:
                    film_list.append(trans_sensitive(title.strip().split('\t')[1]))
            with open(root + bte, 'r', encoding=encoding) as f:
                for title in f:
                    book_list_real.append(title.strip())
            # with open(root + 'p_' + bte, 'r', encoding=encoding) as f:
            #     for title in f:
                    book_list.append(trans_sensitive(title.strip().split('\t')[1]))
            i = 0
            num = 0
            count = 0
            len_m = len(film_list)
            len_b = len(book_list)
            with open(root + 'film_book.txt', 'r', encoding=encoding) as f:
                for line in f:
                    num, count = join(line, len_m, film_list, film_list_real,
                                      len_b, book_list, book_list_real, count, num, f_left, f_join)
                    i += 1
                    print(i, num)
            print('link_num:', num)
            print('pair_num:', count)
    # with open(root + 'book_set.txt', 'w+', encoding=encoding) as f0:
    #     with open(root + 'movie_set.txt', 'w+', encoding=encoding) as f1:
    #         books = []
    #         movies = []
    #         for l in link:
    #             if l[0] not in books:
    #                 books.append(l[0])
    #                 f0.write(l[0])
    #             if l[1] not in movies:
    #                 movies.append(l[1])
    #                 f1.write(l[1])


if __name__ == '__main__':

    # You can preprocess first and output local files to reduce time.
    # Codes commented out in full_join() in should be changed.

    # process_entity()

    # You can also change codes commented out in match() and ceiling() for the rule of matching and filtering.
    full_join()

    # TODO: use multiprocessing to speed up.
