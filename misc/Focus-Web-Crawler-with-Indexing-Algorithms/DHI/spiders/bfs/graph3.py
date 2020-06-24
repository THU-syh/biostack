import sqlite3
import networkx as nx
import matplotlib.pyplot as plt



conn = sqlite3.connect('C:/Users/HP/Dropbox/Sem 8/CAT401/Project coding/DHI/DHI/spiders/bfs/bfs.db')
print "Opened database successfully";

nodes = conn.execute("SELECT DEPTH from BFS")
edges = conn.execute("SELECT LINKS from BFS")

G = nx.Graph()

nx.add_path(G, nodes)

nx.draw(G,with_labels=True)
plt.show()
