"""
    Visualizer for de Bruijn graphs. Contains logic for dispaying
    a specific layer of a de Bruijn "onion".
    Author: Benjamin Keefer
    Version: November 12th, 2024
"""
import networkx as nx
import matplotlib.pyplot as plt
edges = []
sequence = []
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

def main():
    global k, n
    G = nx.DiGraph()
    # Gathers parameters for the sequence
    lay_val = 2
    n = int(input("Enter n: "))
    k = int(input("Enter k: "))
    lay_bool = str(input("Lay? (y/n) "))
    lay_bool = lay_bool.lower()
    # Generates de Bruijn sequence
    de_bruijn(k, n)
    # Updates sequence according to lay
    if(lay_bool == "y"):
        lay_val = int(input("Enter lay value: "))
        lay(lay_val)
    # Gets edge pairs from sequence
    load_edges()
    # Draws de Bruijn graph
    G.add_edges_from(edges)
    try:
        nx.draw_planar(G, with_labels=True)
    except:
        nx.draw_spring(G, with_labels=True)
    plt.show()

# Generates edge pairs from the de Bruijn sequence
def load_edges():
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
    if(x >= k or x < 1):
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
    # Adds 0^(k-1) to the sequence
    start = []
    for i in range(n):
        start.append(0)
    sub_seq.append(start)
    sequence = sub_seq

if __name__ == "__main__":
    main()
