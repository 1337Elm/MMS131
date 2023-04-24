""" File that completes task 2.3 for assignment 2 in the course: MMS131

    Author: Benjamin Elm Jonsson 20011117 (2023) benjamin.elmjonsson@gmail.com
"""

#--------Imports------------

import numpy as np
import matplotlib
import warnings

#---------------------------
#Ignore deprecation warnings
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

def read_file(filename: str) -> np.array:
    """Function that generates 2d array (row,column) from the text file.
        For this solution the matrix needs to be square i.e (10x10) for example.

        :param filename: str, path to the file

        :param return: np.array, 2d array of the "map"
    """
    file =  open(filename)
    lines = file.readlines()

    matrix = []
    for i in range(len(lines)):
        numbers = []
        for j in range(len(lines[i])):
            try:
                numbers.append(int(lines[i][j]))
            except:
                pass
        matrix.append(numbers)

    return np.array(matrix)


def find_path(matrix: np.array) ->list:
    """Function that implements Best First Search in order to reach target node (end node)
        The function does this by going through all neighbors of the current node and calculates
        the distance to the target. The neighbor that is closest to the target is chosen to be the new current node.
        The previous current node is updated to "visited" and cant be chosen again.
        The function then iterates until the current node is the target node.
        The function then returns the path taken in order to reach the target.

        :param matrix: np.array, 2d array of the "map"

        :param return: np.array, 2d array of each node traversed 
    """
    startNode = np.where(matrix == 2)
    startNode = [int(startNode[0]),int(startNode[1])]

    endNode = np.where(matrix == 3)
    endNode = [int(endNode[0]),int(endNode[1])]

    current_node = startNode
    
    distance = []
    path = []
    path.append(startNode)

    while current_node != endNode:
        neighbors = find_neighbors(matrix,current_node,case = 1)

        if endNode in neighbors:
            current_node = endNode
            path.append(current_node)
        else:
            for i in range(len(neighbors)):
                distance.append(calc_distance(matrix,neighbors[i]))
        
            matrix[current_node[0],current_node[1]] = 5 #5 means cant visit,
            #which is true for already visited nodes, i.e this node has now been visited

            current_node = neighbors[np.where(distance == np.min(distance))] #Update current node to node whish has min distance to target
            path.append(current_node)

    return path


def find_neighbors(matrix: np.array, node: list,case: int) -> list:
    """ Function that finds valid neighbors for a given node. It checks which connectivity to use
        (4 or 8). Then for each positivly indexed neighbor (no looping around) checks if this number is 0 or 3
        i.e if the node is traversable. If this is true it is added to a list of valid neighbors which is then returned.

        :param matrix: np.array, 2d array of the "map"
        :param node: list, the coordinates for the node
        :param case: int, determines if 4 or 8 connectivity is to be used

        :param return: np.array, 2d array of coordinates for valid neighbors
    """
    neighbors = []
    i = int(node[0])
    j = int(node[1])

    if case == 1:
        combinations = [[i+1,j],[i-1,j],[i,j+1],[i,j-1]]
    else:
        combinations = [[i+1,j],[i-1,j],[i,j+1],[i,j-1],[i+1,j+1],[i-1,j-1],[i+1,j-1],[i-1,j+1]]

    for i in combinations:
        try:
            if i[0] < 0 or i[1] < 0:
                pass
            else:
                val = matrix[i[0],i[1]]
            
                if val == 0 or val == 3:
                    neighbors.append([i[0],i[1]])
        except:
            pass
    
    return neighbors


def calc_distance(matrix: np.array, currentNode: list) -> int:
    pass


def main():
    filename = "maze_small.txt"

    matrix = read_file("Assignment2/2.4/" + filename)
    print(matrix)
    
    find_path(matrix)

    print(matrix)


if __name__ == '__main__':
    main()