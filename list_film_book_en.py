import re

import bs4
import requests
from bs4 import BeautifulSoup
from opencc import OpenCC


encoding = 'utf-8'
domin_name = 'https://en.wikipedia.org'
href1 = '/wiki/Category:Lists_of_films_based_on_books'
api_metadata = 'https://en.wikipedia.org/api/rest_v1/page/metadata/'
api_summary = 'https://en.wikipedia.org/api/rest_v1/page/summary/'
film_list = []
pair_dic = {}
catogories = []

PAD_TOKEN = '<!GETNONE>'

s = requests.session()
s.keep_alive = False

f = open('data/film_book.txt', 'w+', encoding='utf-8')
# ffc = open('data/film_category.txt', 'a', encoding='utf-8')
# fbc = open('data/book_category.txt', 'a', encoding='utf-8')

stop = ['', '?', '(?)', 'uncredited', '(uncredited)', '?(uncredited)']

def dispose(name):
    if name in stop:
        return ''
    name = re.sub('[♠♦*]', '', name)
    match = re.match('^"([^"]+)"$', name)
    if match:
        name = match.group(1)
    return name.strip()


def get_title(td):
    if td.string:
        return td.string
    i = td.find('i')
    if i:
        name = i.string if i.string else i.get_text()
        return name
    a = td.find('a')
    if a:
        name = a.string if a.string else a.get_text()
        if td.get_text().startswith(name):
            return name
    return ''


def get_urltoken(a, title):
    if a and 'href' in a.attrs:
        return a.attrs['href'][6:]
    else:
        return re.sub(' ', '_', title)


def search(href):
    """
    extract data from tables.
    """
    try:
        r = s.get(domin_name + href, timeout=30)
        r.encoding = encoding
        soup = BeautifulSoup(r.text, 'html.parser')
    except:
        print("Network err!")

    wikitables = soup.find_all('table', {'class':'wikitable'})
    for table in wikitables:
        pair = ['', '']
        i, j = 0, 0
        for th in table.find_all('th'):
            if th.get_text() == 'Film':
                break
            i += 1
        for th in table.find_all('th'):
            if th.get_text() == 'Sourcework' or th.get_text() == 'Source work':
                break
            j += 1

        skip = 0
        for tr in table.find_all('tr')[1:]:

            if skip > 0:
                tds = tr.find_all('td')
                pair[1] = dispose(get_title(tds[0]))
                if pair[0] != '' and pair[1] != '':
                    # pair_dic[pair[0]] = pair[1]
                    print(pair[0], '\t', pair[1])
                    f.write('\t'.join(pair) + '\n')
                    # get_category(tds[0].find('a'), pair[1], fbc)
                skip -= 1
                continue
            tds = tr.find_all('td')
            if tds is None:
                continue
            if 'rowspan' in tds[0].attrs:
                skip = int(tds[0].attrs['rowspan'])-1

            if len(tds) <= i or len(tds) <= j:
                continue

            pair[0] = dispose(get_title(tds[i]))
            pair[1] = dispose(get_title(tds[j]))
            if pair[0] != '' and pair[1] != '':
                # pair_dic[pair[0]] = pair[1]
                print(pair[0], '\t', pair[1])
                f.write('\t'.join(pair) + '\n')
                # get_category(tds[i].find('a'), pair[0], ffc)
                # get_category(tds[j].find('a'), pair[1], fbc)


def search2(href):
    try:
        r = s.get(domin_name + href, timeout=30)
        r.encoding = encoding
        soup = BeautifulSoup(r.text, 'html.parser')
    except:
        print("Network err!")

    wikitables = soup.find_all('table', {'class':'wikitable'})
    for table in wikitables:
        for tr in table.find_all('tr')[1:]:
            tds = tr.find_all('td')
            book = dispose(get_title(tds[0]))
            # get_category(tds[0].find('a'), book, fbc)
            if book:
                for a in tds[1].find_all('a'):
                    if a.string:
                        print(a.string, '\t', book)
                        f.write(a.string + '\t' + book + '\n')
                        # get_category(a, a.string, ffc)


def get_category(a, title, f):
    """
    Get categories from metadata.
    """
    url_token = get_urltoken(a, title)
    r = s.get(api_metadata + url_token)
    try:
        json = r.json()
        for category in json['categories']:
            f.write(title + '\t' + category['titles']['display'] + '\n')
    except:
        url_token = re.sub(' ', '_', title)
        r = s.get(api_metadata + url_token)
        try:
            json = r.json()
            for category in json['categories']:
                f.write(title + '\t' + category['titles']['display'] + '\n')
        except Exception as e2:
            print(str(e2))



def crawl(href):
    """
    Get all category entrances and then use the method `search` for each.
    Specially, *List of children's books made into feature films* need the method `search2` for its different structure.
    """
    try:
        r = s.get(domin_name + href, timeout=30)
        r.encoding = encoding
        soup = BeautifulSoup(r.text, 'html.parser')
    except:
        print("Network err!")
    categories_soup = soup.find('div', {'class': 'mw-category'})
    href_list = []
    for category in categories_soup.find_all('a'):
        if category.string in ['List of children\'s books made into feature films',
                               'Lists of book-based war films']:
            continue
        href_list.append(category.attrs['href'])

    for href in href_list:
        print(href)
        search(href)

    search2('/wiki/List_of_children%27s_books_made_into_feature_films')




def main():
    try:
        crawl(href1)
    finally:
        if f:
            f.flush()
            f.close()
        # if ffc:
        #     ffc.flush()
        #     ffc.close()
        # if fbc:
        #     fbc.flush()
        #     fbc.close()



if __name__ == '__main__':
    main()