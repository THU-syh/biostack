import sqlite3
import networkx as nx
import matplotlib.pyplot as plt

conn = sqlite3.connect('C:/Users/HP/Dropbox/Sem 8/CAT401/Project coding/DHI/DHI/spiders/bfs/bfs.db')
print "Opened database successfully";


nodes = conn.execute("SELECT DEPTH from BFS")
G = nx.Graph()

G= nx.complete_graph(nodes)

nx.draw(G,with_labels=True)
plt.show()
