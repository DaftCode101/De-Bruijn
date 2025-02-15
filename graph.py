import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math as m
import time as t

"""
    Visualizer for de Bruijn graphs. Contains functions for dispaying
    a specific (onion) layer of a de Bruijn graph as well as efficiently
    computing number of hamiltonian paths in that layer.
    Author: Benjamin Keefer
    Version: February 14th, 2024
"""
edges = []
sequence = []
adjacency = {}
G = nx.DiGraph()
startTime = 0
endTime = 0
num = 0
k = 2
n = 2

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
    s = [0 for i in range(n)]
    for i in range(k ** n):
        sequence.append(s)
        s = shift(s)

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
                edges.append(("".join(str(num) for num in word), "".join(str(num) for num in other_word)))
    sequence = edges

# Updates sequence to only contain a single layer of the "onion"
def lay(x : int):
    global sequence
    # Adds words in lay to seq
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
    global sequence
    for edge in sequence:
        if(edge[0] in adjacency):
            adjacency[edge[0]].append(edge[1])
        else:
            adjacency[edge[0]] = [edge[1]]

# TODO: Write a backtracing algorithm to efficiently compute # of hamiltonian paths
def num_of_paths_backtracing(start, end, print_paths, start_path) -> int:
    pass

# TODO: Determine if a shift construction of hamiltonian paths is possible
def num_of_paths_rotations(start, end) -> int:
    pass

# Computes the number of hamiltonian paths in a layer of an onion de Bruijn sequence
# If print_paths is True it will print those paths, otherwise it won't
def num_of_paths(start, end, print_paths) -> int:
    global num
    idx = 0
    all_paths = []
    
    visited = {}
    for edge in list(adjacency.keys()):
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
        for neighbor in adjacency[current_vertex]:
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

# TODO: Occ(n, k, r) graph where k-1 occurs r times
# Hypothesis: L(Occ(n, k, r)) = Occ(n+1, k, r+1)
def occurences(r) -> list:
    global sequence
    removed = []

    for word in edges:
        occ_1 = 0
        occ_2 = 0
        split = list(word[0])
        split2 = list(word[1])
        for i in range(len(split)):
            if(split[i] == str(k - 1)):
                occ_1 += 1
        for i in range(len(split2)):
            if(split2[i] == str(k - 1)):
                occ_2 += 1
        if occ_1 == r and occ_2 == r:
            removed.append(word)
        # if not (word[0] == zero or word[1] == zero):
        #     removed.append(word)

    sequence = removed
    return removed

def line_graph() -> list:
    global sequence
    new_edges = []
    for i in range(len(sequence)):
        for j in range(len(sequence)):
            if(sequence[i][1] == sequence[j][0]):
                new_edges.append((sequence[i][0][1:] + "" + sequence[i][1][-1], sequence[i][1][1:] + "" + sequence[j][1][-1]))
        
    sequence = new_edges
    unique()

# Prints whether or not this graph is Eulerian
def graph_is_eulerian():
    global G
    in_degrees = list(G.in_degree)
    out_degrees = list(G.out_degree)
    eulerian = True
    for i in range(len(in_degrees)):
        if(in_degrees[i][0] != out_degrees[i][0]):
            print("Problem, this shouldn't happen")
            return
        if(in_degrees[i][1] != out_degrees[i][1]):
            eulerian = False
    if(eulerian):
        print(f"Line(Lay({n}, {k})) is eulerian")
    else:
        print(f"Line(Lay({n}, {k})) is NOT eulerian")
    return

# Prints how many hamiltonian paths exist in the layer of the de Bruijn graph
def find_hamiltonian_paths(lay_val):
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

    print("start: " + start)
    print("end: " + end)
    startTimer()

    paths = num_of_paths(start, end, False)

    endTimer()
    print(dTime())
    print("Number of hamiltonian paths: " + str(paths))
    print("Number of vertices: " + str(len(adjacency)))
    return

# Constructs and displays de Bruijn graph
def show_graph():
    global G

    try:
        nx.draw_planar(G, with_labels=True)
    except:
        nx.draw_spring(G, with_labels=True)

    plt.show()
    return

def directed_incidence_matrix():
    pass

# Computes number of hamiltonian paths in Lay(n, k) in polynomial time
def number_of_hamiltonian_paths(show_steps):
    global sequence, G, n
    
    # Gets the vertices in the graph
    vertices = [v for v in G.nodes()]
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
        # degrees[edge[1]] = degrees[edge[1]] + 1 # uncomment to change to vertex degree
        

    # Constructs degree matrix
    deg = [[0 for _ in range(l_v)] for _ in range(l_v)]
    i = 0
    for v in vertices:
        deg[i][i] = degrees[v]
        i += 1
    
    # Constructs Laplacian matrix
    laplacian = [[(deg[i][j] - adj[i][j]) for j in range(l_v)] for i in range(l_v)]
    sub_arrays = produce_sub_arrays(laplacian)

    # Computes eigenvalues
    eigenvalues = np.linalg.eigvals(np.array(laplacian))
    show_eigenvalues = False

    def print_line(l):
        print("=" * l)

    # Prints the vertices, the adjacency matrix, the degree matrix,
    # and the laplacian matrix
    if show_steps:
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

    
    if show_steps:
        print("Determinants (# of arborescences):")
    i = 0
    dets = []
    for arr in sub_arrays:
        det = round(np.linalg.det(np.array(arr)))
        if show_steps:
            print(f"Vertex: {vertices[i]} Determinant: {det}")
            print(np.array(arr))
        
        dets.append(det)
        i += 1
        if(i > 1):
            if(dets[i - 1] != det):
                print("Not all # of arborescences are the same, problem!")

    if show_steps:
        show_laplacian_determinant = False
        if show_laplacian_determinant:
            print_line(l)
            print(f"Det(Laplacian(Line(Lay({n}, {k})))) = {np.linalg.det(np.array(laplacian))}")

        if show_eigenvalues:
            print("Eigenvalues:")
            for val in eigenvalues:
                if(val.imag > 0.0):
                    print(f"{val.real} + {val.imag}i")
                elif(val.imag < 0.0):
                    print(f"{val.real} - {abs(val.imag)}i")
                else:
                    print(f"{val.real}")
        print_line(l)
        print(f"Number of arborescences: {dets[0]}")
        print_line(l)
    
    degree_product = 0
    i = 0
    for v in vertices:
        if i == 0:
            degree_product = m.factorial(degrees[v] - 1)
            i += 1
        else:
            degree_product *= m.factorial(degrees[v] - 1)
            i += 1

    if show_steps:
        print(f"# of paths = number of arborescences * product((degree(v) - 1)! where v is a vertex in the Lay({n}, {k})) graph")
        print_line(l)
    print(f"Number of hamiltonian paths in Line(Lay({n}, {k})): {dets[0] * degree_product}")
    return

# Removes duplicate edges from the sequence
def unique():
    global sequence
    new_sequence = []
    s = set()
    for seq in sequence:
        if not s.__contains__(seq):
            s.add(seq)
            new_sequence.append(seq)
    sequence = new_sequence
    return

# Generates sub arrays such that the ith row and ith column are removed
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

# Main method
def main():
    global k, n, sequence
    
    # Gathers parameters for the sequence
    n = int(input("Enter n: "))
    k = int(input("Enter k: "))
    lay_val = k - 1
    lay_bool = str(input("Lay? (y/n) "))
    lay_bool = lay_bool.lower()

    # Generates de Bruijn sequence
    de_bruijn(k, n)

    # Updates sequence according to lay
    if(lay_bool == "y"):
        lay(lay_val)

    # Constructs layer of the de Bruijn graph
    load_edges()
    remove_zero_edges()
    unique()
    adjacency_lists()
    line_graph()

    
    graph_is_eulerian()
    
    G.add_edges_from(sequence)
    # find_hamiltonian_paths(lay_val) # O(v!)

    # Find number of hamiltonian paths O(?)
    number_of_hamiltonian_paths(False) # change False to True for printing computational steps
    # show_graph()
    return

# Calls main
if __name__ == "__main__":
    try:
        main()
    except:
        print("Invalid input")
