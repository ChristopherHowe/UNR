import networkx as nx;
import matplotlib.pyplot as plt;
from depthFirstSearch import depthFirstSearch;

def isBipartite(graph: nx.Graph):
    is_bipartite = True
    
    def bipartiteHelper(vertex, graph: nx.Graph):
        nonlocal is_bipartite
        neighborsState = None
        for neighbor in graph.adj[vertex]: 
            if graph.nodes[neighbor]['color'] != 'white': # check if neighbor has been explored
                if neighborsState != graph.nodes[neighbor]['set'] and neighborsState != None:
                    is_bipartite = False
                else:
                    neighborsState = graph.nodes[neighbor]['set']            
        
        if neighborsState != None:
            graph.nodes[vertex]['set'] = not neighborsState
        else:
            graph.nodes[vertex]['set'] = True
        
    depthFirstSearch(graph, bipartiteHelper)
    return is_bipartite
