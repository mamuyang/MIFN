import html
import re
import nltk
from nltk import word_tokenize, pos_tag

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



# process HTML chars, $ and !
def trans_html(word):
    # if re.search('&[#\w]+;', word):
    #     for key in dic.keys():
    #         word = re.sub(key, dic[key], word)
    word = html.unescape(word)
    res = re.search('!+(\w[^!]+\w)!+', word)
    if res:
        word = re.sub('!+(\w[^!]+\w)!+', r'\1', word)
    if not re.search('\${3,}', word):
        word = re.sub('\$\$([^A-Z\d])', r'ss\1', word)
        word = re.sub('\$([^A-Z\d])', r's\1', word)
        word = re.sub('([A-Za-z])\$\$', r'\1ss', word)
        word = re.sub('([A-Za-z])\$', r'\1s', word)
    return word


# process sensitive words and *
def trans_sensitive(title):
    if re.search('[^ \W\d][^ \-\'.\w][^ \W\d]', title):
        title = re.sub('sh*t', 'shit', title, flags=re.I)
        title = re.sub('f[u*][*c]k', 'fuck', title, flags=re.I)
        title = re.sub('m*a*s*h', 'mash', title, flags=re.I)
        title = re.sub('\*tm|\*r', '', title)
    title = re.sub('([^ \W\d])[^ \-\'.\w]([^ \W\d])', r'\1 \2', title)
    title = re.sub(' \*(\w)', r'\1', title)
    return title


# flag=0: special for movie
def process(title, flag):
    # title = re.sub('[Tt]he ', '', title)
    # title = re.sub('[Aa]n? ', '', title)
    title = re.sub('\(.*vol(ume)?[. :/\'\-\d)].*$', '', title, flags=re.I)
    title = re.sub(r' *[-/\[\\:,] *vol(ume)?[. :/\-\d)].*$', '', title, flags=re.I)
    title = re.sub(' *vol(ume)?[., :/\-\d].*$', '', title, flags=re.I)
    if flag == 0:
        title = re.sub('\(.*VHS.*\).*$', '', title, flags=re.I)
        title = re.sub('\[? *VHS *\]? *.*$', '', title, flags=re.I)
        title = re.sub(':.*DVD.*$', '', title, flags=re.I)
        title = re.sub(' *DVD *', '', title, flags=re.I)
    # title = re.sub('\(.+\) *$', '', title)
    title = title.strip('&-# ."\'')
    # title = ' '.join(lemmatize_and_stem_all(title))
    return title


mte = 'movie_title_entity.txt'
bte = 'book_title_entity.txt'
encoding = 'utf-8'


def process_entity(flag):
    entity = mte if flag == 0 else bte
    with open('data/' + entity, 'r', encoding=encoding) as f1:
        with open('data/disposed/' + entity, 'w+', encoding=encoding) as f2:
            for line in f1:
                title = line.strip().split('\t')[1]
                title = process(title, 0)
                f2.write(title + '\n')


def sort_title(flag):
    entity = mte if flag == 0 else bte
    with open('data/disposed/' + entity, 'r', encoding=encoding) as f:
        with open('data/disposed/sorted/' + mte, 'w+', encoding=encoding) as w:
            list = [process(line.strip(), flag=flag) for line in f]
            list.sort()
            for l in list:
                w.write(l + '\n')


def shorten(title):
    title = re.sub(':.+$', '', title)
    title = re.sub(' [\d\W]+$', '', title)
    # title = re.sub('[Tt]he ', '', title)
    # title = re.sub('[Aa]n ', '', title)
    title = title.strip(' "\'')
    return title



def cut(title):
    title = ' '.join(lemmatize_and_stem_all(shorten(title)))
    return title
