import re

import requests
from bs4 import BeautifulSoup

encoding = 'utf-8'
domin_name = 'https://en.wikipedia.org'
href1 = '/wiki/Category:Television_shows_based_on_books'
film_list = []
pair_dic = []
catogories = []

PAD_TOKEN = '<!GETNONE>'

s = requests.session()
s.keep_alive = False

f = open('data/film_book.txt', 'a', encoding='utf-8')
# tf = open('dataset/0a/film_head.txt', 'w+', encoding='utf-8')

def cut(str):
    return re.sub('|\(.*book.*\)'
                  '|\(.*film.*\)'
                  '|\(.*series.*\)', '', str, flags=re.I).strip()


def search_infobox(infobox):
    # infobox = soup.find('table', {'class': 'infobox vevent'})
    pair1 = ['', '']
    if infobox:
        th = infobox.find('th')
        if th.string:
            pair1[0] = th.string

        origin = infobox.find('th', string=re.compile('based', flags=re.I))
        if origin:
            td = origin.parent.find('td')
            if td:
                if td.string:
                    pair1[1] = td.string
                else:
                    text = td.get_text()
                    a = td.find('a')
                    if a and a.string and text.startswith(a.string):
                        pair1[1] = a.string
                    else:
                        i = td.find('i')
                        if i and i.string and text.startswith(i.string):
                            pair1[1] = i.string
                        else:
                            match = re.match('(.+)(Written )?by .+', text, flags=re.I)
                            if match:
                                pair1[1] = match.group(1)
                pair1[1] = re.sub(' *\( *novel *\) *', '', pair1[1]).strip()
                match = re.search('[Bb]ased( on)? *(.+)$', pair1[1])
                if match:
                    pair1[1] = match.group(2)
                if pair1[1] == '':
                    pair1[1] = text

    return pair1


def search(href):
    try:
        r = s.get(domin_name + href)
        r.encoding = encoding
        soup = BeautifulSoup(r.text, 'html.parser')
    except:
        print("Network err!", href)
        return

    # film_title  file_title_en  book_title  book_title_en
    pair = ['', '']
    pair1 = ['', '']
    pair2 = ['', '']

    title = soup.find('title').string
    title = re.sub(' - Wikipedia', '', title)
    # 从词条表获取中英名，如有，获得原著名
    # 如果没有电影的括号注释，先查找是否有电影一栏
    match = re.match('^(.+)(\(.*series.*\))$', title)
    title_name = cut(title)
    found = False
    if match:
        film_columns = soup.find_all(
            'span', {'class':'mw-headline'},
            string=re.compile('series'))
        # for fc in film_columns:
        #     tf.write(fc.string+'\n')
        if film_columns:
            found = True
    film_text = ''
    fp = None
    if found:
        for film_column in film_columns:
            infobox = film_column.parent.find_next_sibling('table', {'class': 'infobox vevent'})
            pair = search_infobox(infobox)
            fp = film_column.find_next('p')
            if fp and title_name in fp.get_text():
                film_text = fp.get_text()
    else:
        infobox = soup.find('table', {'class': 'infobox vevent'})
        pair1 = search_infobox(infobox)

    # 从第一段简介获取，可信度较低
    # introduction = film_text if found else soup.find('p').get_text()
    cursor = soup.find('p')
    for i in range(10):
        if cursor is None or title_name in cursor.get_text():
            break
        cursor = cursor.find_next('p')
    introduction = ''
    if found:
        introduction = film_text
        cursor = fp
    elif cursor:
        introduction = cursor.get_text()
    first_p = None
    first_para = ''
    bs = soup.find_all('b', string=title_name)
    if bs:
        for b in bs:
            if b.parent and b.parent.name == 'p':
                first_p = b.parent
                break
    if introduction == '' or introduction == '\n' or introduction is None:
        first_para = first_p.get_text() if first_p else ''
        introduction = first_para
        cursor = first_p

    pair2[0] = title_name

    same_name = False
    if re.search('based on.+same name', introduction, flags=re.I):
        pair2[1] = pair2[0]

    # 如果有电影一栏，并且词条不包含电影两字，极有可能是一个大词条，主栏可能是书籍
    pair3 = ['', '']
    if found and not re.search('series', title):
        searchObj = re.search('[Bb]ook|[Nn]ovel|[Ww]ritten', first_para)
        if searchObj:
            pair3[1] = title_name



    for i in range(2):
        pair[i] = pair1[i] if pair1[i] else pair2[i]
    pair[1] = pair[1] if pair[1] else pair3[1]

    if pair[0] and pair[0] not in film_list and pair[1]:
        film_list.append(pair[0])
        print(pair)
        f.write("\t".join(pair)+'\n')


def crawl(href, get_children=True):
    """
    First, crawl subcategories recursively.
    Second, crawl pages in category below.
    """
    try:
        r = s.get(domin_name + href, timeout=30)
        r.encoding = encoding
        soup = BeautifulSoup(r.text, 'html.parser')
    except:
        print("Network err!", href)
    h2 = soup.find('h2', string=re.compile('Subcategories'))
    if h2:
        children = h2.find_next('div', {'class': 'mw-content-ltr'}).find_all('div', {'class': 'CategoryTreeSection'})
        if get_children and children:
            for child in children:
                a = child.find('a')
                print(a.string)
                catogory = a.string
                if catogory in catogories or re.search('^User:|^Template:', catogory):
                    continue
                catogories.append(catogory)
                crawl(a.attrs['href'])
    # pages = soup.find('div', {'class':'mw-category'})
    # if not pages:
    if h2 and children:
        pages = children[-1].parent.find_next('div', {'class': 'mw-content-ltr'})
    else:
        h2 = soup.find('h2', string=re.compile('Pages'))
        if h2:
            pages = h2.find_next('div', {'class': 'mw-content-ltr'})
        else:
            pages = None
    if pages:
        for a in pages.find_all('a'):
            if re.search('^User:|^Template:', a.string):
                continue
            # try:
            search(a.attrs['href'])
            # except:
            #     print('产生异常', a.attrs['href'])


def main():
    try:
        crawl(href1)
    finally:
        if f:
            f.flush()
            f.close()


if __name__ == '__main__':
    main()