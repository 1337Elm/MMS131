""" File that completes task 2.3 for assignment 2 in the course: MMS131

    Author: Benjamin Elm Jonsson 20011117 (2023) benjamin.elmjonsson@gmail.com
"""

#--------Imports------------

import numpy as np
import matplotlib
import warnings
import sys

#---------------------------
#Ignore deprecation warnings
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

def read_file(filename: str) -> np.array:
    file =  open(filename)
    lines = file.readlines()

    numbers = []
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            try:
                numbers.append(int((lines[i][j])))
            except:
                pass

    matrix = []
    for i in range(0,len(numbers),len(lines)):
        row = []
        for j in range(len(lines[0])-1):
            row.append(numbers[i + j])

        matrix.append(row)

    return np.array(matrix)


def find_path(matrix: np.array) ->list:
    startNode = np.where(matrix == 2)
    endNode = np.where(matrix == 3)
    current_node = startNode
    
    distance = []
    path = []
    path.append(startNode)

    while current_node != endNode:
        neighbors = find_neighbors(matrix,current_node,case = 1)

        for i in range(len(neighbors)):
            distance.append(calc_distance(matrix,neighbors[i]))
        
        matrix[current_node[0],current_node[1]] = 1 #1 means cant visit,
        #which is true for already visited nodes, i.e this node has now been visited

        current_node = neighbors[np.where(distance == np.min(distance))] #Update current node to node whish has min distance to target
        path.append(current_node)

    path.append(endNode)
    return path


def find_neighbors(matrix: np.array, node: list,case: int) -> list:
    neighbors = []
    i = int(node[0])
    j = int(node[1])

    if case == 1:
        combinations = [[i+1,j],[i-1,j],[i,j+1],[i,j-1]]
    else:
        combinations = [[i+1,j],[i-1,j],[i,j+1],[i,j-1],[i+1,j+1],[i-1,j-1],[i+1,j-1],[i-1,j+1]]

    for i in combinations:
        try:
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
    
    print(find_path(matrix))


if __name__ == '__main__':
    main()