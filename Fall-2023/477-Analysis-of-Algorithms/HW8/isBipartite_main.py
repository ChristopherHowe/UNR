import networkx as nx
import matplotlib.pyplot as plt
from isBipartite_alg import isBipartite

def main():
    bipartite = nx.Graph()  # Create a graph
    bipartite.add_nodes_from(['x1', 'x2', 'x3', 'y1', 'y2','y3']) # Add nodes
    bipartite.add_edges_from([('x1', 'y1'), ('x1', 'y2'), ('x2', 'y1'), ('x2', 'y2'), ('x2', 'y3'),('x3','y2'),('x3','y3'),]) # Add edges
    #nx.draw(bipartite, with_labels=True, font_weight='bold') # Visualize the graph
    #plt.show()

    print(f"This graph is {isBipartite(bipartite)}")

    not_bipartite = nx.Graph()  # Create a graph
    not_bipartite.add_nodes_from(['a','b','c','d'])
    not_bipartite.add_edges_from([('a','b'),('a','c'),('b','c'),('c','d'),('b','d')])
    #nx.draw(not_bipartite, with_labels=True, font_weight='bold') # Visualize the graph
    #plt.show()

    topo = nx.DiGraph()  # Create a graph
    topo.add_nodes_from(['a','b','c','d','e','f','g'])
    topo.add_edges_from([
        ('a','b'),('a','c'),('b','g'),('b','e'),
        ('c','f'),('d','a'),('d','b'),('d','c'),
        ('d','f'),('d','g'),('g','e'),('g','f')
    ])
    nx.draw(topo, with_labels=True, font_weight='bold') # Visualize the graph
    plt.show()


    print(f"This graph is {isBipartite(topo)}")
    print("Graph Attributes:")
    print("Nodes:", topo.nodes(data=True))
    print("Edges:", topo.edges(data=True))

    # Print node attributes
    print("\nNode Attributes:")
    for node, attrs in topo.nodes(data=True):
        print(f"Node {node}: {attrs}")

main()
