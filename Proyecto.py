import networkx as nx
import matplotlib.pyplot as plt
from math import inf


def dibujaGrafos(E):
    aristas = []
    aristas_peso = []
    nodos = []
    for i in E:
        valor = E.get(i)
        for j in E.get(i):
            aristas.append((i,j))
            nodos.append(i)
            nodos.append(j)
            aristas_peso.append((i,j,valor.get(j)))
            
            
    #print(aristas)
    #print(aristas_peso)
            
    
    G = nx.DiGraph()
    #G.add_edges_from(aristas)
    
    for i in aristas_peso:
        G.add_edge(i[0], i[1], weight = i[2])
    
    val_map = {'a': 1.0,
                'b': 0.5714285714285714,
                'c': 0.0
                }
    
    values = [val_map.get(node, 0.25) for node in G.nodes()]
    
    edge_colours = ['black' for edge in G.edges()]
    black_edges = [edge for edge in G.edges()]
    
    labels = nx.get_edge_attributes(G, "weight")
    
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color = 'blue', node_size = 500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
    

def dijkstra2(E,start,end):
    S = E
    L = {}
    Pred = {}
    camino = []
    
    for i in S:
        L[i] = inf
    L[start] = 0
    B = L.copy()
    #NodA = set()
    
    while len(S) > 0:

        nodo =[x for x in B.keys() if L[x]==min(B.values())][0]
        
        #nodo = [x for x in L.keys() if L[x]==  min(set(L.values()) - NodA) and x in S][0]
        #NodA.add(nodo)
        
        for edge, weight in E[nodo].items():
            comp = weight + L[nodo]
            if comp < L[edge]:
                L[edge] = comp
                B[edge] = comp
                Pred[edge] = nodo
        S.pop(nodo)
        B.pop(nodo)
    
    ubicacion = end
    
    while ubicacion != start:
        try:
            camino.append(ubicacion)
            ubicacion = Pred[ubicacion]
        except KeyError:
            print("No existe un camino del nodo " + start + " a el nodo " + end )
            break
    camino.append(start)
    camino = camino[::-1]
    if L[end] != inf:
         return camino, L[end]
    
    else:
        camino = []
        return camino, inf


def dibujaCamino(P):
    E_resultado = {}
    for i in range (0,len(P) - 1):
        E_resultado[P[i]] = P[i+1] 
    
    
    aristas = []
    aristas_peso = []
    nodos = []
    for i in E_resultado:
        aristas.append((i,E_resultado.get(i)))
            #aristas_peso.append((i,j,valor.get(j)))
            
    #print(aristas_peso)
            
    
    G = nx.DiGraph()
    G.add_edges_from(aristas)
    
    # for i in aristas_peso:
    #     G.add_edge(i[0], i[1], weight = i[2])
    
    val_map = {'a': 1.0,
                'b': 0.5714285714285714,
                'c': 0.0
                }
    
    values = [val_map.get(node, 0.25) for node in G.nodes()]
    
    edge_colours = ['black' for edge in G.edges()]
    black_edges = [edge for edge in G.edges()]
    
    #labels = nx.get_edge_attributes(G, "weight")
    
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color = 'blue', node_size = 500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=True)
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def dijkstraColors(E,Colors, start, objective):
    S = E
    L = {}
    Pred = {}
    camino = []
    
    for i in S:
        L[i] = inf
    L[start] = 0
    B = L.copy()
    #NodA = set()
    
    while len(S) > 0:

        nodo =[x for x in B.keys() if L[x]==min(B.values())][0]
        
        #nodo = [x for x in L.keys() if L[x]==  min(set(L.values()) - NodA) and x in S][0]
        #NodA.add(nodo)
        
        for edge, weight in E[nodo].items():
            comp = weight + L[nodo]
            if comp < L[edge]:
                L[edge] = comp
                B[edge] = comp
                Pred[edge] = nodo
        S.pop(nodo)
        B.pop(nodo)
    
    valores = L.copy()
    #print(valores)
    if Colors.get(start) == objective:
        print("¡Ya estás en el objetivo!")
        return None, None
    else:
        for i in L:
            if Colors.get(i) != objective:
                valores.pop(i)
        
        if len(valores) == 0:
            print("Lo sentimos, no encontramos lo que solicitas en esta región.")
            return None, None
        else:
            end = list(valores.keys())[list(valores.values()).index(min(valores.values()))]
    
   # destino = end.copy()
    ubicacion = end
    print(ubicacion)
    while ubicacion != start:
        try:
            camino.append(ubicacion)
            ubicacion = Pred[ubicacion]
        except KeyError:
            print("No existe un camino del nodo " + start + " a el nodo " + end )
            #camino = []
            break
    camino.append(start)
    camino = camino[::-1]
    if L[end] != inf:
         # print('Shortest distance is ' + str(shortest_distance[goal]))
         # print('And the path is ' + str(path))
         return camino, L[end]
    
    else:
        camino = []
        return camino, inf


def dibujaColores(E,Colors, P):
    aristas = []
    aristas_peso = []
    nodos = []
    for i in E:
        valor = E.get(i)
        for j in E.get(i):
            aristas.append((i,j))
            nodos.append(i)
            nodos.append(j)
            aristas_peso.append((i,j,valor.get(j)))
            
    
            
    
    # print(aristas)
    # print(aristas_peso)
    # print(nodos)
    
    G = nx.DiGraph()
    #G.add_edges_from(aristas)
    
    for i in aristas_peso:
        G.add_edge(i[0], i[1], weight = i[2])
    
    val_map = {'a': 1.0,
                'b': 0.5714285714285714,
                'c': 0.0
                }
    
    color_map = []
    for node in G:
        #print(Colors.get(node))
        color_map.append(Colors.get(node))
        
    #values = [val_map.get(node, 0.25) for node in G.nodes()]
    
    edge_colours = ['black' for edge in G.edges()]
    black_edges = [edge for edge in G.edges()]
    
    labels = nx.get_edge_attributes(G, "weight")
    colors = nx.get_node_attributes(G, "color")
    
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color = color_map, node_size = 500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

    
    E_resultado = {}
    for i in range (0,len(P) - 1):
        E_resultado[P[i]] = P[i+1] 
    
    
    aristas = []
    aristas_peso = []
    nodos = []
    for i in E_resultado:
        aristas.append((i,E_resultado.get(i)))
            #aristas_peso.append((i,j,valor.get(j)))
            
    #print(aristas_peso)
            
    
    G = nx.DiGraph()
    G.add_edges_from(aristas)
    
    # for i in aristas_peso:
    #     G.add_edge(i[0], i[1], weight = i[2])
    
    val_map = {'a': 1.0,
                'b': 0.5714285714285714,
                'c': 0.0
                }
    
    color_map = []
    for node in G:
        #print(Colors.get(node))
        color_map.append(Colors.get(node))
        
    values = [val_map.get(node, 0.25) for node in G.nodes()]
    
    edge_colours = ['black' for edge in G.edges()]
    black_edges = [edge for edge in G.edges()]
    
    #labels = nx.get_edge_attributes(G, "weight")
    
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color = color_map, node_size = 500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=True)
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def dijkstraConParada(E,start,parada,end):
    
    E_orig = E.copy()
    a, b = dijkstra2(E, start, parada)
    c, d = dijkstra2(E_orig, parada, end)
    c.pop(0)
    r = a+c
    return r


todos=[]
def encuentra_caminos(G,start,end,primera,camino):
    nodo=start
    camino.append(nodo)
    #if primera:
    #    camino.append(nodo)
    #    primera=0
    if nodo==end:
        #print("Camino: ",camino)
        aux=tuple(camino.copy())
        todos.append(aux)
        #print("soy todos",todos)
        camino.pop()
        #todos.append(camino)
        #camino = []
        return aux
    for edge, weight in G[nodo].items():
        #print(f"soy {camino} con el nodo {nodo} y la arista {edge}")
        x=encuentra_caminos(G,edge,end,0,camino)
    camino.pop()
    #print("pase de aca")
    #print(todos)
    return todos

def ford_ful(graph_flujo,todos):
    tolerancias = {}
    respuesta = 0
    #c=[['a','b','c','d','e'],['a','c','d']]
    pairs=[]
    pairs_t=[]
    for i in todos:
        for j in range(len(i)-1):
            pairs_t.append([i[j],i[j+1]])
        pairs.append(pairs_t)
        pairs_t = []
    
    seguir = True
    #print(max(tolerancias))
    while True:
        escogido = []
        for i in pairs:
            candidatos = []
            for j in i:
                for a in graph_flujo:
                    if a[0] == j[0] and a[1] == j[1]:
                        candidatos.append(a[2].get('capacity') - a[2].get('flow'))
            escogido.append(min(candidatos))
        #print(escogido)
            
        for i in range (0,len(todos)):
            tolerancias[todos[i]] = escogido[i] 
            
        z = list(tolerancias.keys())[list(tolerancias.values()).index(max(tolerancias.values()))]
        
        #print(tolerancias.get(z))
        
        respuesta += tolerancias.get(z)
        
        if tolerancias.get(z) == 0:
            #seguir = False
            break
            
        E_resultado = []
        for i in range (0,len(z) - 1):
            E_resultado.append([z[i],z[i+1]])
        #print(E_resultado)
        
        for i in E_resultado:
                for a in graph_flujo:
                    if a[0] == i[0] and a[1] == i[1]:
                        a[2]['flow'] = a[2]['flow'] + tolerancias.get(z)
                        # print(a[2].get('capacity'))
                        # print(a[2].get('flow'))
                        
                        
        #print(graph_flujo)        
        #print()
    return "El flujo máximo de su grafo es: " + str(respuesta)



""" Aquí ejemplos de Dijsktra normal """


# graph = {'a':{'b':10,'c':3},'b':{'c':1,'d':2},'c':{'d':8,'e':2},'d':{'e':7},'e':{'d':9}}

# E = {"a": {"b": 94, "e": 89}, "b":{"c": 91}, "c":{"d":98, "g": 90}, "d":{}, "e":{"f": 95, "i": 89}, "f":{"b": 91, "g": 92},"g":{"h": 91, "k":92}, "h":{"d": 87},
# "i":{"j":89, "m": 92 }, "j":{"f":90, "k": 96}, "k":{"l": 91, "o": 83}, "l": {"h":93}, "m":{"n":97}, "n":{"j": 89, "o": 87}, "o":{"p":95}, "p":{"l":92}}

# a, b = dijkstra2(E, 'a', 'k')
# print(b)
# dibujaCamino(a)


"""Ejemplo con parada"""

# graph2 = {'a':{'b':10,'c':3},'b':{'c':1,'d':2},'c':{'d':8,'f':5},'d':{'e':7}, 'f':{'e': 4},'e':{}}

# dibujaGrafos(graph2)

# a = dijkstraConParada(graph2,'a','c','e')

# dibujaCamino(a)

"""Aquí ejemplos de colores"""

# #graph2 = {'a':{'b':10,'c':3},'b':{'c':1,'d':2},'c':{'d':8,'e':2},'d':{'e':7},'e':{}}

# graph2 = {'a':{'b':10,'c':3},'b':{'c':1,'d':2},'c':{'d':8,'f':5},'d':{'e':7}, 'f':{'e': 4},'e':{}}

# Color_graph2 = {'a': 'blue', 'b': 'purple', 'c': 'yellow', 'd': 'purple', 'e': 'red', 'f':'red'}

# a,b = dijkstraColors(graph2.copy(), Color_graph2, 'a', 'red')


# #graph2 = {'a':{'b':10,'c':3},'b':{'c':1,'d':2},'c':{'d':8,'e':2},'d':{'e':7},'e':{'d':9}}


# dibujaColores(graph2, Color_graph2, a)


# #a = dijkstraConParada(graph2.copy(), 'a', 'b', 'e')

# #dibujaColores(graph2, Color_graph2, a)


# #grafo_latex = {"a": {"b":4,"c":8},"b": {"c": 2}}
# #Color_grafo_latex = {'a': 'blue', 'b': 'purple', 'c': 'yellow'}

# #a = []
# #dibujaColores(grafo_latex, Color_grafo_latex, a)



"""Aqui ejemplo de todos los caminos"""
# graph2 = {'a':{'b':10,'c':3},'b':{'c':1,'d':2},'c':{'d':8,'f':5},'d':{'e':7}, 'f':{'e': 4},'e':{}}
# dibujaGrafos(graph2)
# z = encuentra_caminos(graph2, 'a','e', 1, [])

# for i in z:
#     dibujaCamino(i)
"""Aquí ejemplos de flujo"""



# graph2 = {'a':{'b':10,'c':3},'b':{'c':1,'d':2},'c':{'d':8,'f':5},'d':{'e':7}, 'f':{'e': 4},'e':{}}
# dibujaGrafos(graph2)
# z = encuentra_caminos(graph2, 'a','e', 1, [])

# graph_flujo = [
#     ('a', 'b', {'capacity': 4, 'flow': 0}),
#     ('a', 'c', {'capacity': 5, 'flow': 0}),
#     ('b', 'c', {'capacity': 7, 'flow': 0}),
#     ('b', 'd', {'capacity': 5, 'flow': 0}),
#     ('c', 'd', {'capacity': 4, 'flow': 0}),
#     ('c', 'f', {'capacity': 8, 'flow': 0}),
#     ('d', 'e', {'capacity': 3, 'flow': 0}),
#     ('f', 'e', {'capacity': 6, 'flow': 0})
# ]

# print(ford_ful(graph_flujo,z))
    
    


