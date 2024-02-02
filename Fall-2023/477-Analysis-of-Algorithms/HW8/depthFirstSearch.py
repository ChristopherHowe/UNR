import networkx as nx;

def depthFirstSearch(graph:nx.Graph, func):
    t = 0
    vertSet = graph.nodes()
    for vertex in vertSet:
        graph.nodes[vertex]['color']='white'
        graph.nodes[vertex]['pi']=''

    for vertex in vertSet:
        if graph.nodes[vertex]['color'] == 'white':
            t = dfs_visit(vertex, graph, t, func)

def dfs_visit(vertex, graph: nx.Graph, t, func):
    func(vertex, graph)
    graph.nodes[vertex]['color'] = 'grey'
    t += 1
    graph.nodes[vertex]['d'] = t
    for v in graph.adj[vertex]:
        if graph.nodes[v]['color'] == 'white':
            graph.nodes[vertex]['pi']=vertex
            t = dfs_visit(v, graph, t, func)
    graph.nodes[vertex]['color'] = 'black'
    t += 1
    graph.nodes[vertex]['f'] = t
    return t
