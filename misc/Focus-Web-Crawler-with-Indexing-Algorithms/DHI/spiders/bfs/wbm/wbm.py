import re
import urllib2
from sets import Set
import matplotlib.pyplot as plt
import sqlite3

conn = sqlite3.connect('C:/Users/HP/Dropbox/Sem 8/CAT401/Project coding/DHI/DHI/spiders/bfs/bfs.db')

conn.execute('''CREATE TABLE BFS
         (LINKS           TEXT    NOT NULL,
         DEPTH            INT     NOT NULL);''')

print "Table created successfully";
start_link ='http://www.westernbotanicalmedicine.com/links.html'

urls = Set([start_link])

def findId(source):
    l = re.findall(r'"(http[s]*://\S+)"',source)
    return l

def get_source(url):
    response = urllib2.urlopen(url)
    page_source = response.read()
    return page_source

    conn.execute("INSERT INTO BFS(LINKS, DEPTH) \
          VALUES ( ?, ?)", ( source, depth));

    conn.commit()


def search(source, depth):
    if depth==4:
        return
    print source, depth

    try:
        page_source = get_source(source)
        links = Set(findId(page_source))
    except:
        print 'some error encountered'
        return

    global urls
    for link in links:
        if link not in urls:
            urls = urls|Set([link])        

    for link in urls:
        search(link,depth+1)

search(start_link,0)
conn.close()
