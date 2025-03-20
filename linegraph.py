"""
Practice math research coding problem for Jewan.
TODO Implement line_graph function.

Author: Benjamin Keefer
Version: March 19th, 2025
"""

# Example inputs of de bruijn graphs.
def test_line_graph():
    graph1 = [('100', '010'), ('100', '110'), ('010', '101'), ('010', '001'), ('101', '010'), 
              ('101', '110'), ('110', '111'), ('110', '011'), ('111', '111'), ('111', '011'), 
              ('011', '101'), ('011', '001'), ('001', '100')]

    graph2 = [('200', '120'), ('200', '020'), ('200', '220'), ('120', '012'), ('120', '112'), 
              ('120', '212'), ('012', '201'), ('201', '120'), ('201', '020'), ('201', '220'),
              ('020', '102'), ('020', '202'), ('020', '002'), ('102', '210'), ('210', '121'), 
              ('210', '021'), ('210', '221'), ('121', '012'), ('121', '112'), ('121', '212'), 
              ('112', '211'), ('211', '121'), ('211', '021'), ('211', '221'), ('021', '102'), 
              ('021', '202'), ('021', '002'), ('202', '120'), ('202', '020'), ('202', '220'), 
              ('220', '122'), ('220', '222'), ('220', '022'), ('122', '012'), ('122', '112'), 
              ('122', '212'), ('212', '121'), ('212', '021'), ('212', '221'), ('221', '122'), 
              ('221', '222'), ('221', '022'), ('222', '122'), ('222', '222'), ('222', '022'), 
              ('022', '102'), ('022', '202'), ('022', '002'), ('002', '200')]
    
    # Displays the line graphs as constructed in the line_graph function.
    print("Line graph of DB(3, 2)")
    print(line_graph(graph1))
    print("Line graph of DB(3, 3)")
    print(line_graph(graph2))

"""
This function takes in a de bruijn graph as its argument and outputs its line graph.
The graph is a list of edges represented as tuples containing the words in DB(n, k) as strings.
Inside each tuple is a word of length n over an alphabet k:
each edge is of the form (x(tao), (phi)x).

We denote the line graph of DB(n, k) as LDB(n, k).
For an edge of the form (x(tao), (phi)x), the corresponding vertex
in the line graph of DB(n, k) is of the form (phi)x(tao).
For other notational clarification read the Onion paper.

Definition of line graph: https://en.wikipedia.org/wiki/Line_graph
"""
def line_graph(graph : list) -> list:
    lineGraph = [] # should contain tuples of strings

    # TODO: Add the edges of the LDB(n, k) before it is returned.

    return lineGraph

# Calls the function to test the line_graph method.
if __name__ == "__main__":
    test_line_graph()
