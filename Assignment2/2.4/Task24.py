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
    path = []
    startNode = np.where(matrix == 2)
    endNode = np.where(matrix == 3)

    neighbors = find_neighbors(matrix,startNode)

    return path


def calc_distance(matrix: np.array, currentNode: list) -> int:
    pass


def find_neighbors(matrix: np.array, node: list) -> list:
    pass


def main():
    filename = "maze_small.txt"

    matrix = read_file("Assignment2/2.4/" + filename)
    print(matrix)
    find_path(matrix)



if __name__ == '__main__':
    main()