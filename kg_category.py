import re

import Levenshtein
import pandas as pd
import requests
import wikipedia
from util import *
from join import exist_match, match
import multiprocessing
import numpy as np


def transform(s):
    return trans_sensitive(shorten(process(trans_html(s), 0)))


pattern_note = re.compile(' *\(.+\) *$| *[Tt]he *| *[Aa]n? *')


def delete_note(s):
    return re.sub(pattern_note, '', s)


def clip(type):
    print(type)
    # entity_set = pd.read_csv('data/output/' + type + '_entity_dict.txt', delimiter='\t', header=None)[0].tolist()
    df = pd.read_csv('data/id2title/' + type + '_entity_title.txt', delimiter='\t', header=None)
    # df = df[df[0].isin(entity_set)]
    df[1] = df[1].apply(transform)
    df.to_csv('data/output/' + type + '_entity_title.txt', sep='\t', header=False, index=False)


trash = []
entities = '(movie|film|book|drama|video|television|novel|comic|play|series|show|anime|manga|ova|tv)s?'
entity_pattern = re.compile(entities)
stop_pattern = re.compile(entities + '$')
dirty_pattern = re.compile('cs1|article|set in|shot in|error|wiki|template|use|games|albums|radio'
                     '|page|using|usage|language|link|adapted|adaption|unfinished|introduced')


def valid(category):
    if not category:
        return False
    if re.match('\W+$', category):
        return False
    s = category.lower()
    if re.match(stop_pattern, s):
        return False
    if category in trash:
        return False
    if re.search(dirty_pattern, s) is not None:
        trash.append(category)
        return False
    return True


def extract(category):
    category = re.sub('\d{4}s?|\d{2}th[ \-][Cc]entury', '', category).strip()
    category = re.sub(' {2,}', ' ', category)
    if category is None:
        return category
    res = re.search('(about|by|based on|shot at) (.+)', category)
    if res:
        category = res.group(2)
    res = re.search('(.+) (' + entities + ')+',
                category, flags=re.I)
    if res:
        category = res.group(1)
    return category


delete_word = re.compile('Americans?|Canadian|British|Hong Kong|Chinese|Italian|Australian'
                         'French|Japanese|Spanish|African|English|the United States|Indian|'
                         'and|[Dd]ebuts?|\(genre\)|Western|(South )?Korean|German|'
                         + entities)


def is_entity(categories):
    has_entity = False
    for c in categories:
        res = re.search('(ers|[Aa]ctors?|ists)', c)
        if res:
            back = res.span()[1]
            if len(c) == back or c[back] == ' ':
                break
        if re.search(entity_pattern, c):
            has_entity = True
            break
    return has_entity


def wash_func(categories):
    categories = eval(categories)
    if not is_entity(categories):
        return []
    categories = [re.sub(delete_word, '', extract(c)).strip().strip('-') for c in categories]
    categories = [re.sub(' {2,}', '', c).lower() for c in categories if valid(c)]
    categories = list(set(categories))
    return categories



def ceiling(query, title):
    if query.count(' ') <= 2:
        res = re.search(re.escape(query), title, flags=re.I)
        if res:
            if re.match(' [^\w,\']|( *$)', title[res.span()[1]:]):
                if res.span()[0] == 0:
                    return True
                elif res.span()[0] >= 2 \
                        and re.match('[^\w,\']{2}', title[res.span()[0] - 2:res.span()[0]]):
                    return True
        return False
    return True


api_metadata = 'https://en.wikipedia.org/api/rest_v1/page/metadata/'

cnt = 0


def get_category(title):
    global cnt
    cnt += 1
    print(cnt)
    try:
        for res in wikipedia.search(title, results=2):
            # print(title, '\t', res)
            s1 = delete_note(title)
            s2 = delete_note(res)
            if match(s1, s2, True) or Levenshtein.jaro_winkler(s1, s2) > 0.8:
                r = s.get(api_metadata + re.sub(' ', '_', res))
                try:
                    json = r.json()
                    categories = [c['titles']['display'] for c in json['categories']]
                    if not categories and is_entity(categories): continue
                    print(categories)
                    return [c for c in categories if valid(c)]
                except:
                    continue
    except Exception as e:
        print(str(e))
        return []
    return []


def task(df, type, bs=200):
    i = 0
    while i < len(df):
        sub = df[i: i+bs].copy()
        sub['category'] = sub['title'].apply(get_category)
        sub.to_csv('data/output/' + type + '_df.csv', index=False, header=False, mode='a')
        i += bs


def generate_category(type='movie'):
    df = pd.read_csv('data/output/' + type + '_entity_title.txt', delimiter='\t', header=None)
    df.columns = ['id', 'title']
    # over = pd.read_csv('data/output/' + type + '_df.csv', header=None)
    # ids = over[0].unique()
    # empty = over[over[2] == '[]'][0].unique()
    # df = df[~df['id'].isin(ids) or df['id'].isin(empty)]
    # df = df[~df['id'].isin(ids)]
    # del over
    # del ids
    # del empty
    sum = len(df)
    print(sum, sum//n_process)
    for i in range(n_process):
        p = multiprocessing.Process(target=task,
                                    args=(df[i * sum//n_process:
                                             (i+1) * sum//n_process], type, ))
        p.start()


def wash(type='movie'):
    df = pd.read_csv('data/output/' + type + '_df.csv', header=None)
    df.columns = ['id', 'title', 'category']
    df = df[df['category'] != '[]']
    df['category'] = df['category'].apply(wash_func)
    df = df[df['category'].map(len) > 0]
    # df.to_csv('data/output/' + type + '_df1.csv', header=False, index=False)
    df2 = pd.DataFrame({'id': df.id.repeat(df.category.str.len()),
                       'category': np.concatenate(df.category.values)})
    df_count = df2['category'].value_counts()
    hot = df_count[df_count.values > 150].index
    tags = set(hot[hot.map(lambda x: len(x.split(' '))) == 1].to_list()) - {'in'}
    def split_tags(x):
        x = x.strip()
        s = set(x.split(' '))
        if 2 <= len(s) <= 3:
            common = s & tags
            if s != common:
                common.add(x)
            return list(common)
        else:
            return [x]
    df2['category'] = df2['category'].apply(split_tags)
    df2 = pd.DataFrame({'id': df2.id.repeat(df2.category.str.len()),
                       'category': np.concatenate(df2.category.values)})
    # df_kg = pd.read_csv('data/output/' + type + '_entity_dict.txt', header=None, delimiter='\t')
    df3 = pd.read_csv('data/output/' + type + '_raw_categories.txt', header=None, delimiter='\t')
    df3.columns = ['id', 'category']
    # df3 = df3[df3.id.isin(df_kg[0].values)]
    df2 = df2.append(df3, ignore_index=True).drop_duplicates().sort_values(by='id')
    df2.to_csv('data/output/' + type + '_id_category.csv', sep='\t', header=False, index=False)


def output():
    # output relation of kg index and category id
    df1_dict = pd.read_csv('data/output/movie_rid2index.txt', header=None, delimiter='\t', names=['id', 'eid'])
    df2_dict = pd.read_csv('data/output/book_rid2index.txt', header=None, delimiter='\t', names=['id', 'eid'])

    df1 = pd.read_csv('data/output/movie_id_category.csv', header=None, delimiter='\t')
    df1.columns = ['id', 'category']
    df2 = pd.read_csv('data/output/book_id_category.csv', header=None, delimiter='\t')
    df2.columns = ['id', 'category']

    category = df1.append(df2)['category'].value_counts().reset_index()
    category = category[['index']]
    category.columns = ['category']
    category['cid'] = category.index
    category.to_csv('data/output/category_dict.txt', sep='\t', header=False, index=False)

    df1 = df1.merge(category, on='category')
    df1 = df1.merge(df1_dict, on='id').sort_values(by='eid')
    df1[['eid', 'cid']].to_csv('data/output/movie_id_cid.txt', sep='\t', header=False, index=False)
    df2 = df2.merge(category, on='category')
    df2 = df2.merge(df2_dict, on='id').sort_values(by='eid')
    df2[['eid', 'cid']].to_csv('data/output/book_id_cid.txt', sep='\t', header=False, index=False)


def rid2index():
    for type in ['movie', 'book']:
        df = pd.read_csv('data/output/' + type + '_entity_dict.txt', header=None, delimiter='\t', names=['id', 'eid'])
        df2 = pd.read_csv('data/output/' + 'entity_id2index_' + type + '.txt', header=None, delimiter='\t', names=['eid', 'eidx'])
        if type == 'B':
            df2['eid'] -= 158410
        df = df.merge(df2).sort_values('eidx')
        df[['id', 'eidx']].to_csv('data/output/' + type + '_rid2index.txt', header=False, sep='\t', index=False)


s = requests.session()
s.keep_alive = False

if __name__ == '__main__':

    n_process = 5
    for type in ['movie', 'book']:
        clip(type)
        generate_category(type)
        wash('book')

    # output()




