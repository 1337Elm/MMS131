""" File that completes task 2.3 for assignment 2 in the course: MMS131

    Author: Benjamin Elm Jonsson 20011117 (2023) benjamin.elmjonsson@gmail.com
"""

#--------Imports------------

import numpy as np
import random

#---------------------------

def read_file(filename: str) -> np.array:
    """Function that generates 2d array (row,column) from the text file.
        For this solution the matrix needs to be square i.e (10x10) for example.

        :param filename: str, path to the file

        :param return: np.array, 2d array of the "map"
    """
    file =  open(filename)
    lines = file.readlines()

    #Go through each row, append the row to the matrix 
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


def find_path(matrix: np.array,case: int) ->list:
    """Function that implements Best First Search in order to reach target node (end node)
        The function does this by going through all neighbors of the current node and calculates
        the distance to the target. The neighbor that is closest to the target is chosen to be the new current node.
        The previous current node is updated to "visited" and cant be chosen again.
        The function then iterates until the current node is the target node.
        The function then returns the path taken in order to reach the target.

        :param matrix: np.array, 2d array of the "map"
        :param case: int, choosing 4- or 8-connectivity

        :param return: np.array, 2d array of each node traversed 
    """
    #Find index of start and end node
    startNode = np.where(matrix == 2)
    startNode = [int(startNode[0]),int(startNode[1])]

    endNode = np.where(matrix == 3)
    endNode = [int(endNode[0]),int(endNode[1])]

    #Set starting node as the first node and initialize a best distance above 0 to the end node
    current_node = startNode
    best_dist = 1
    while best_dist > 0:
        #Find neighbors and set current node to visited (+5)
        distance = []
        neighbors = find_neighbors(matrix,current_node,case)
        matrix[current_node[0],current_node[1]] = matrix[current_node[0],current_node[1]] + 5 #+5 means visited

        #Compute distance for each neighbor
        for i in range(len(neighbors)):
            distance.append(calc_distance(matrix,neighbors[i]))

        #Find best neighbor and set this node as new current node
        #If the best distance is 0, this is the end node, set end node to visited and break
        best_dist = np.min(distance)
        if best_dist == 0:
            current_node = endNode
            matrix[current_node[0],current_node[1]] = matrix[current_node[0],current_node[1]] + 5 #+5 means visited
            break
        else:
            try:
                best_neighbor = neighbors[int(np.where(distance == best_dist)[0])]
            except:
                
                #Possible solution to always pick the best neighbor, if more than 1 have
                #euqally good distances, but this isnt included in a greedy BFS hence it is commented
                
                # new_distance1 = []
                # new_distance2 = []
                # new_neighbors1 = find_neighbors(matrix,neighbors[int(np.where(distance == best_dist)[0][0])],case)
                # for i in range(len(new_neighbors1)):
                #     new_distance1.append(calc_distance(matrix,new_neighbors1[i]))
                
                # new_neighbors2 = find_neighbors(matrix,neighbors[int(np.where(distance == best_dist)[0][1])],case)
                # for i in range(len(new_neighbors2)):
                #     new_distance2.append(calc_distance(matrix,new_neighbors2[i]))

                # if np.min(new_distance1) > np.min(new_distance2):
                #     best_neighbor = neighbors[np.where(distance == best_dist)[0][1]]
                # else:
                #     best_neighbor = neighbors[np.where(distance == best_dist)[0][0]]

                best_neighbor = neighbors[np.where(distance == best_dist)[0][0]]

            current_node = [best_neighbor[0],best_neighbor[1]] #Update current node to node whish has min distance to target

    return matrix


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
    y = node[0]
    x = node[1]

    #All possible neighbors depending on case
    if case == 1:
        combinations = [[y+1,x],[y-1,x],[y,x+1],[y,x-1]]
    else:
        combinations = [[y+1,x],[y-1,x],[y,x+1],[y,x-1],[y+1,x+1],[y-1,x-1],[y+1,x-1],[y-1,x+1]]

    for i in combinations:
        #Try lets me ignore if a neighbor is in reality outside of the map
        #This would give an error but now it is simply ignored
        try:
            #If negative, it loops around and its no longer the correct square -> no negative values
            if i[0] < 0 or i[1] < 0:
                pass
            else:
                #Only values of 0 or 3 are available to choose, if the value isnt one of these pass
                val = matrix[i[0],i[1]]
            
                if val == 0 or val == 3:
                    neighbors.append([i[0],i[1]])
        except:
            pass
    
    return neighbors


def calc_distance(matrix: np.array, currentNode: list) -> int:
    """Function that calculates the manhattan distance from the current node to the end node.

        :param matrix: np.array, 2d array of the "map"
        :param currentNode: list, coordinates for the current node

        :param return: int, manhattan distance from current node to the end node
    """
    endNode = np.where(matrix == 3)
    endNode = [int(endNode[0]),int(endNode[1])]

    return ((np.abs(currentNode[0] - endNode[0]))**2 + (np.abs(currentNode[1]-endNode[1]))**2)**(1/2)


def main():
    """ Main function of the script. Chooses which maze to solve and which case (4 or 8-connectivity)
        Then saves the solution to a textfile
    """
    filename = "maze_big.txt"

    case = 1
    matrix = read_file("Assignment2/2.4/" + filename)    
    new_matrix = find_path(matrix,case)

    if filename == "maze_big.txt":
        np.savetxt("Assignment2/2.4/solutions/solvedMatrix_big" + str(case) + ".txt",new_matrix,fmt = '%d', delimiter= "")
    else:
        np.savetxt("Assignment2/2.4/solutions/solvedMatrix_small" + str(case) + ".txt",new_matrix,fmt = '%d', delimiter= "")
        

if __name__ == '__main__':
    main()