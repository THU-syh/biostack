import networkx as nx
import matplotlib.pyplot as plt
import sqlite3

conn = sqlite3.connect('C:/Users/HP/Dropbox/Sem 8/CAT401/Project coding/DHI/DHI/spiders/bfs/bfs.db')
print "Opened database successfully";


aa= conn.execute("SELECT DEPTH from BFS")
list = [aa]

G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
K3 = nx.Graph([list])
nx.draw(K3,with_labels=True)
plt.show()
