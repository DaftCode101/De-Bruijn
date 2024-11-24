"""
    Visualizer for de Bruijn graphs. Contains logic for dispaying
    a specific layer of a de Bruijn "onion".
    Author: Benjamin Keefer
    Version: November 24th, 2024
"""
import networkx as nx
import matplotlib.pyplot as plt
edges = []
sequence = []
adjacency = {}
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
    edges = []
    for word in sequence:
        for other_word in sequence:
            add = True
            for i in range(len(other_word) - 1):
                if(word[i] != other_word[i+1]):
                    add = False
            if(add):
                edges.append(("".join(str(num) for num in word), "".join(str(num) for num in other_word)))

# Updates sequence to only contain a single layer of the "onion"
def lay(x : int):
    global sequence
    # Checks validity of lay value
    if(x >= k or x < 0):
        print("Invalid lay value")
        return
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
def adjacency_lists(edges):
    for edge in edges:
        if(edge[0] in adjacency):
            adjacency[edge[0]].append(edge[1])
        else:
            adjacency[edge[0]] = [edge[1]]

# Computes the number of hamiltonian paths in a layer of an onion de Bruijn sequence
# If print_paths is True it will print those paths, otherwise it won't
def num_of_paths(start, end, print_paths) -> int:
    global num
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
    return num

def main():
    global k, n
    G = nx.DiGraph()
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

    # Makes layer from the de Bruijn sequence
    load_edges()
    remove_zero_edges()
    adjacency_lists(sequence)

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

    print("Number of hamiltonian paths: " + str(num_of_paths(start, end, False)))

    # Draws de Bruijn graph
    # G.add_edges_from(sequence)
    # try:
    #     nx.draw_planar(G, with_labels=True)
    # except:
    #     nx.draw_spring(G, with_labels=True)
    # plt.show()

if __name__ == "__main__":
    try:
        main()
    except:
        print("Invalid input")
