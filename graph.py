import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math as m
import time as t

"""
    Math research code.
    Contains functions for dispaying a specific (onion) layer of a 
    de Bruijn graph and efficiently computing the number of hamiltonian
    paths in that layer.

    Author: Benjamin Keefer
    Version: June 20th, 2024
"""


# De Bruijn generation code from amirrubin87 on github
def val(s):
    n = len(s)
    return sum([s[- i] * (k ** (i - 1)) for i in range(1, n + 1)])
def val_star(s):
    n = len(s)
    return max([val(s[i:] + s[:i]) for i in range(n)])
def rotate_zeros(s):
    return s if s[0]!=0 else rotate_zeros(s[1:]+s[:1])
def a_dagger(s):
    return s[-1] != 0 and val_star(s)==val(rotate_zeros(s))
def shift(s):
    x = s[:-1]
    sigma = s[-1]
    if sigma < (k - 1) and a_dagger(x + [sigma + 1]):
        return [sigma + 1] + x
    elif a_dagger(x + [sigma]):
        return [0] + x
    else:
        return [sigma] + x
def de_bruijn(k_val: int, n_val: int):
    global k
    global n
    k = k_val
    n = n_val
    s = [0 for _ in range(n)]
    for i in range(k ** n):
        sequence.append(s)
        s = shift(s)

# Extends the alphabet beyond 0-9
def char(x : int):
    if x < 10:
        return str(x)
    elif x < 36:
        return str(chr(x + 87))
    elif x < 62:
        return str(chr(x + 29))
    return str(chr(x + 113))

# Generates edge pairs from the de Bruijn sequence
def load_edges():
    global edges
    global sequence
    edges = []
    for word in sequence:
        for other_word in sequence:
            add = True
            for i in range(len(other_word) - 1):
                if(word[i] != other_word[i+1]):
                    add = False
            if(add):
                edges.append(("".join(char(num) for num in word), "".join(char(num) for num in other_word)))
    sequence = edges

# Updates sequence to only contain a single layer of the "onion"
def lay(x : int):
    global sequence
    sub_seq = []
    for word in sequence:
        in_word = False
        for num in word:
            if(num == x):
                in_word = True
        if(in_word):
            sub_seq.append(word)
    sequence = sub_seq

# Removes edges that start or end with 0^(n) because they're not needed for lay computation
def remove_zero_edges() -> list:
    global sequence
    removed = []
    zero = ""
    for i in range(n):
        zero += str(0)

    for word in edges:
        if not (word[0] == zero or word[1] == zero):
            removed.append(word)

    sequence = removed
    return removed

# Converts list of graph edges into a dictionary of adjacency lists
def adjacency_lists():
    global sequence, ldb
    ldb.clear()
    for edge in sequence:
        if edge[0] not in ldb:
            ldb[edge[0]] = [edge[1]]
        else:
            ldb[edge[0]].append(edge[1])

# Computes the number of hamiltonian paths in a layer of an onion de Bruijn sequence
# If print_paths is True it will print those paths, otherwise it won't
def num_of_paths_greedy(start, end, print_paths) -> int:
    global num
    idx = 0
    all_paths = []
    
    visited = {}
    for edge in list(ldb.keys()):
        visited[edge] = False

    def dfs(current_path, current_vertex, print_paths):
        global num
        if current_vertex == end and len(current_path) == len(visited):
            if(print_paths):
                all_paths.append(current_path[:])
            num = num + 1
            return
        elif current_vertex == end:
            return
        visited[current_vertex] = True
        for neighbor in ldb[current_vertex]:
            if not visited[neighbor]:
                current_path.append(neighbor)
                dfs(current_path, neighbor, print_paths)
                current_path.pop()
        visited[current_vertex] = False
    
    dfs([start], start, print_paths)
    if(print_paths):
        for path in all_paths:
            print(path)
            if(idx % 2 == 1):
                print("=" * len(path) * 6)
            idx += 1
    return num

# Timing functions
def startTimer():
    global startTime
    startTime = t.time()
def endTimer():
    global endTime
    endTime = t.time()
def dTime() -> float:
    return endTime - startTime #the change in time

# Displays de Bruijn graph
def show_graph():
    global G
    G = nx.DiGraph()
    G.add_edges_from(sequence)
    total_vertices = 0
    num_of_ones_in = 0
    num_of_ks_in = 0
    num_of_ones_out = 0
    num_of_ks_out = 0
    for node in G.nodes:
        total_vertices += 1
        indeg = G.in_degree(node)
        outdeg = G.out_degree(node)
        if indeg == 1: num_of_ones_in += 1
        if indeg == k: num_of_ks_in += 1
        if outdeg == 1: num_of_ones_out += 1
        if outdeg == k: num_of_ks_out += 1
        if indeg != outdeg:
            print("Graph is not eulerian!")
    print(f"Number of vertices with in-degree and out-degree of one: {num_of_ones_in}")
    assert(num_of_ones_in == num_of_ones_out)
    assert(num_of_ones_in == ((k-1) ** (n-1)))
    print(f"Number of vertices with in-degree and out-degree of k: {num_of_ks_in}")
    assert(num_of_ks_in == num_of_ks_out)
    assert(num_of_ks_in == ((k ** (n-1)) - ((k-1) ** (n-1))))
    print(f"Total number of vertices: {total_vertices}")
    assert(k ** (n-1) == total_vertices)
    
    try:
        nx.draw_planar(G, with_labels=True)
    except:
        nx.draw_spring(G, with_labels=True)
    plt.show()
    return

def remove_deg_one():
    global G, sequence, vertices
    G.add_edges_from(sequence)
    vertices = []
    for node in G.nodes:
        vertices.append(node)
    in_degrees = list(G.in_degree)
    ver = set()
    for i in range(len(in_degrees)):
        print(in_degrees[i][1])
        if in_degrees[i][1] == 1:
            ver.add(vertices[i])
    removed_sequence = []
    for edge in sequence:
        if edge[0] in ver or edge[1] in ver:
            continue
        removed_sequence.append(edge)
    print(f"Number of vertices with in-deg 1: {len(ver)}")
    sequence = removed_sequence
    G.add_edges_from(sequence)

# Prints whether or not this graph is Eulerian
def graph_is_eulerian(print_bool : bool):
    global G
    G.add_edges_from(sequence)
    global vertices
    vertices = []
    for node in G.nodes:
        vertices.append(node)

    in_degrees = list(G.in_degree)
    out_degrees = list(G.out_degree)

    eulerian = True
    for i in range(len(in_degrees)):
        if(print_bool):
            print(f"Vertex: {vertices[i]} In: {in_degrees[i][1]}, Out: {out_degrees[i][1]}")
        if(in_degrees[i][0] != out_degrees[i][0]):
            raise Exception("Invalid graph")
        if(in_degrees[i][1] != out_degrees[i][1]):
            eulerian = False
    if not print_bool:
        return
    if(eulerian):
        print(f"Kee(Lay({n}, {k})) is eulerian")
    else:
        print(f"Kee(Lay({n}, {k})) is NOT eulerian")
    return

# Prints all hamiltonian paths that exist in the layer of the de Bruijn graph
def find_hamiltonian_paths(lay_val) -> int:
    start_val = lay_val
    end_val = lay_val
    if lay_val == 0:
        start_val = 1
        end_val = k - 1
    start = str(start_val)
    end = ""
    for i in range(n - 1):
        start += "0"
        end += "0"
    end += str(end_val)
    paths = num_of_paths_greedy(start, end, False)
    print(f"Number of hamiltonian cycles in a ({n}, {k}) de Bruijn graph: " + str(paths))
    return paths

# Approximation of determinant using eigenvalues
def eigenvals_approximation(arr) -> int:
    eigenvalues = np.linalg.eigvals(np.array(arr))
    product = 1
    for val in eigenvalues: product *= float(val.real)
    return round(product)

# Computes number of hamiltonian paths in Lay(n, k) in polynomial time
def number_of_hamiltonian_paths(hide_steps):
    global sequence, G, n
    
    # Gets the vertices in the graph
    adjacency_lists()
    vertices = list(ldb.keys())
    l_v = len(vertices)
    l = (l_v * (4 + n))
    
    # Constructs adjacency matrix
    adj = [[0 for _ in range(l_v)] for _ in range(l_v)]
    
    degrees = {}
    for v in vertices:
        degrees[v] = 0

    for edge in sequence:
        adj[vertices.index(edge[0])][vertices.index(edge[1])] = 1
        degrees[edge[1]] = degrees[edge[1]] + 1 # edge[1] for in-degrees, edge[0] for out degrees
        # degrees[edge[0]] = degrees[edge[0]] + 1 # uncomment to change to vertex degree

    # Constructs degree matrix
    deg = [[0 for _ in range(l_v)] for _ in range(l_v)]
    i = 0
    for v in vertices:
        deg[i][i] = degrees[v]
        i += 1
    
    # Constructs Laplacian matrix
    laplacian = [[(deg[i][j] - adj[i][j]) for j in range(l_v)] for i in range(l_v)]
    sub_arrays = produce_sub_arrays(laplacian)

    def print_line(l):
        print("=" * l)

    # Prints the vertices, the adjacency matrix, the degree matrix,
    # and the laplacian matrix
    if not hide_steps:
        print(f"Vertices in the graph: \n{vertices}")
        print_line(l)

        print("Adjacency matrix:")
        for a in adj: print(a)
        print_line(l)

        print("Degree matrix:")
        for d in deg: print(d)
        print_line(l)

        print("Laplacian matrix:")
        for x in laplacian:
            for c in x: print((" " * abs(len(str(c)) - 2)) + str(c), end=" ")
            print()
        print_line(l)

    # Computes and prints number of arborescences for each vertex in the Line(Lay(n, k))
    # This is the left side of the BEST theorem equation
    dets = []
    if hide_steps:
        print(f"Number of arborescences in Kee({n}, {k}): {round(np.linalg.det(np.array(sub_arrays[0])))}")
        dets.append(np.linalg.det(np.array(sub_arrays[0])))
    else:
        print("Determinants (# of arborescences):")
        i = 0
        for arr in sub_arrays:
            det = round(np.linalg.det(np.array(arr)))
            dets.append(det)
            if hide_steps:
                print(f"Vertex: {vertices[i]} Determinant: {dets[0]}")
                print(np.array(arr))
            i += 1
            if(i > 1):
                if(dets[i - 2] != dets[i - 1]):
                    raise Exception("Not all # of arborescences are the same, problem!")
        print_line(l)
        print(f"Number of arborescences: {dets[0]}")
        print_line(l)

    # Prints interesting computations that are unnecessary for hamiltonian path computation
    show_optional = False
    if (not hide_steps) and show_optional:
        print_line(l)
        print(f"Det(Laplacian(Line(Lay({n}, {k})))) = {np.linalg.det(np.array(laplacian))}")
        print("Eigenvalues of the laplacian:")
        eigenvalues = np.linalg.eigvals(np.array(laplacian))
        for val in eigenvalues:
            if(val.imag > 0.0):
                print(f"{val.real} + {val.imag}i")
            elif(val.imag < 0.0):
                print(f"{val.real} - {abs(val.imag)}i")
            else:
                print(f"{val.real}")
        
    # Computes the right side of the BEST theorem equation
    degree_product = 1
    for v in vertices:
        # print(f"{v} contributes {degrees[v] - 1}")
        degree_product *= m.factorial(degrees[v] - 1)

    if not hide_steps:
        print(f"# of paths = number of arborescences * product((degree(v) - 1)! where v is a vertex in the Lay({n}, {k})) graph")
        print_line(l)

    if not (dets[0] % 1 > 0.999 or dets[0] % 1 < 0.001):
        print(dets[0])
        print(dets[0] % 1)
        raise Exception("Determinant computation has floating point value so number of paths computation can't be relied on!")
    else:
        dets[0] = round(dets[0])
    
    num_of_paths = dets[0] * degree_product

    try:
        assert(num_of_paths == (((m.factorial(k))**((k**(n-1))-((k-1)**(n-1))))) / (k**(n-1)))
    except:
        print(num_of_paths)
        print(int(((m.factorial(k))**((k**(n-1))-((k-1)**(n-1))))) / (k**(n-1)))
        raise Exception("Floating point in determinate computation breaking equivalence")

    print(f"Number of eulerian cycles in Kee({n}, {k}) = {num_of_paths}")
    if num_of_paths > 0 and int(m.log10(num_of_paths)+1) > 6:
        print(f"or\napproximately {num_of_paths / (10 ** int(m.log10(num_of_paths))):.1f}e+{int(m.log10(num_of_paths))}")
    return

# Removes duplicate edges from a graph
def unique(graph, swap) -> list:
    new_sequence = []
    s = set()

    for seq in graph:
        if not s.__contains__(seq):
            s.add(seq)
            new_sequence.append(seq)

    if swap:
        global sequence
        sequence = new_sequence

    return new_sequence
    
# Generates all sub arrays where the ith row and ith column are removed
def produce_sub_arrays(arr) -> list:
    sub = []
    for i in range(len(arr)):
        sub_arr = []
        for j in range(0, i):
            x = []
            for s in range(0, i):
                x.append(arr[j][s])
            for s in range(i + 1, len(arr)):
                x.append(arr[j][s])
            sub_arr.append(x)
        for j in range(i + 1, len(arr)):
            x = []
            for s in range(0, i):
                x.append(arr[j][s])
            for s in range(i + 1, len(arr)):
                x.append(arr[j][s])
            sub_arr.append(x)
        sub.append(sub_arr)
    return sub

# Returns the vertices of the given graph
def get_vertices(graph : list) -> list:
    vertices = set()
    for i in range(2):
        for j in range(len(graph)):
            vertices.add(graph[j][i])
    return list(vertices)

"""
Constructs the Key Graph of LDB(n, k) where LDB(n, k) = ldb is
a dictionary that maps each vertex in LDB(n, k) to a list of all
vertices that it has an edge going out to.

Returns Key(LDB(n, k)) as a set of edges.
"""
def key() -> set:
    key = set()
    labeled_vertices = dict()
    pk = 0 # arbitrary use of primary key to label vertices

    for neighborhood in ldb.items():
        vertex = neighborhood[0]

        # Labels current vertex if it has not been labeled yet  
        if not labeled_vertices.__contains__(vertex):
            labeled_vertices[vertex] = str(pk)
            pk += 1
        
        # Labels the adjacent vertices if they have not been labeled yet
        pk_used = False
        for adjacent_vertex in neighborhood[1]:
            if not labeled_vertices.__contains__(adjacent_vertex):
                labeled_vertices[adjacent_vertex] = pk
                pk_used = True

        """
        Adds an edge between the current vertex and the 
        vertex of the last-labeled adjacent vertex.
        """
        key.add((str(labeled_vertices[vertex]),\
                 str(labeled_vertices[adjacent_vertex])))

        # Increments labeling key
        if pk_used:
            pk += 1

    global sequence
    sequence = key
    print(f"Key has {len(key)} edges")
    assert(len(key) == ((k**n)-((k-1)**n)))

    # Returns Key(LDB(n, k)) represented as a set of edges.
    return key

def degrees(new_k : int):
    global n, k
    k = new_k
    print(f"n: {n}, k: {k}")
    lay_val = k - 1
    de_bruijn(k, n)

    # Updates sequence according to lay

    lay(lay_val)
    # Constructs layer of the de Bruijn graph
    load_edges()
    
    remove_zero_edges()
    unique(sequence, True)
    adjacency_lists()
    # find_hamiltonian_paths(lay_val) # O(v!)
    # kee()
    

    print_edges = False
    if print_edges:
        for w in sequence:
                print(w)

    # find_hamiltonian_paths(lay_val) # O(v!)
    
    key()
    number_of_hamiltonian_paths(True)

    # if print_edges:
    #     for w in sequence:
    #         print(w)

    show_graph()

def reset_state():
    global n, k, edges, ldb, vertices, sequence
    global startTime, endTime, num
    edges = []
    sequence = []
    ldb = {}
    vertices = []
    G = nx.DiGraph()
    startTime = 0
    endTime = 0
    num = 0
    # Adjust initial n and k here
    n = 3
    k = 3

# Main method
def main():
    reset_state()
    global k
    for i in range(k, k + 3):
        reset_state()
        degrees(i)
    return

if __name__ == "__main__":
    main()
