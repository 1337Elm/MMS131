""" File that completes task 2.2 for assignment 2 in the course: MMS131

    Author: Benjamin Elm Jonsson 20011117 (2023) benjamin.elmjonsson@gmail.com
"""

#--------Imports------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings

#---------------------------
#Ignore deprecation warnings
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


def read_file(filename: str) -> np.array:
    """ Function that reads the given textfile and returns numpy arrays of 
        the x and y coordinates

        :param filename: string of file to be opened
        :param return: 2 numpy arrays containing x and y data
    """
    
    #Reading given textfile
    with open(filename) as f:
        x = []
        y = []

        #Parse the data and append each x and y coordinate to a list
        for i,line in enumerate(f):
            row = line.split(" ")

            x.append(float(row[0]))
            y.append(float(row[1]))

    f.close()
            
    return np.array(x), np.array(y)


def lloydsAlgo(k: int,guess_x: list,guess_y: list,x: np.array,y: np.array) -> np.array:
    """ Function that finds the final position of the centroids of k clusters using starting guesses
    for x and y. Also saves which centroid is closest for each data point and saves this list to 
    the textfile 'kmeans_label.txt'
    
        :param k: int, number of clusters
        :param guess_x: list, list of starting x-coordinates for the k centroids
        :param guess_y: list, list of starting y-coordinates for the k centroids
        :param x: numpy array, array of x-coordinates for the data points
        :param y: numpy array, array of y-coordinates for the data points

        :return final_pos: numpy array, 2d array of final x and y coordinates for the centroids
        :return closest_centroid: numpy array, array of the closest centroid for the current data point
    """
    #Define limit and starting point
    lim = 0.001
    centroid_pos = [guess_x,guess_y]

    #Initiate lists that are needed later
    dist = []
    closest_centroid = np.zeros(len(x))
    new_posx = np.zeros(k)
    new_posy = np.zeros(k)

    #Variables used to know when to stop the loop
    movement = 0
    final_count = 0

    #Lists for final positions
    final_posx = []
    final_posy = []

    #Loop until all centroids have fulfilled the requirement movement < lim
    while final_count <= 3:
        #Loop through all points and find which centroid is the closest
        for i in range(len(x)):
            for j in range(k):
                dist.append(np.abs(x[i]-centroid_pos[0][j]) +np.abs(y[i] - centroid_pos[1][j]))

            closest_centroid[i] = np.argmin(dist)
            dist.clear()

        #Update centroid position and calculate the movement its then done
        for i in range(k):
            new_posx[i]= np.median(x[closest_centroid == i])
            new_posy[i] = np.median(y[closest_centroid == i])

            movement = np.abs(new_posx[i] - centroid_pos[0][i]) + np.abs(new_posy[i] - centroid_pos[1][i])

            #Check if criteria is fullfilled, if not update position
            #If it is add 1 to the counter, when the counter reaches 3 all final positions are added
            if movement < lim:
                    final_count += 1
                    if final_count == 3:
                        final_posx.append(new_posx)
                        final_posy.append(new_posy)
            else:
                centroid_pos[0][i] = new_posx[i]
                centroid_pos[1][i] = new_posy[i]


    #Add one to every line (needed because indexing begins at 0)
    #And save this to "kmeans_label.txt" file
    closest_centroid = closest_centroid + np.ones(len(x))
    np.savetxt("Assignment2/2.2/kmeans_labels.txt", closest_centroid, fmt = '%d')
    
    #Return final position and list of classification
    return np.array([final_posx,final_posy]), np.array(closest_centroid)


def kNN(point: list, k, i: int, closest_centroid: np.array, x: np.array,y: np.array) -> int:
    """Functions that determines the class of a point
    by majority vote of its k closest neighbors classes

        :param point: list, coordinates for the new data point
        :param k: int/list, number of neighbors wished to be taken in to a count
        :param i: int, the dimension for the L^i norm wished to be used
        :param closest_centroid: numpy array, array of which centroid is closest for the current data point
        :param x: numpy array, array of x-coordinates for all data points
        :param y: numpy array, array of y-coordinates for all data points

        :return voted_class: int, the closest centroid to the new point
    """
    #Initiate variables that are needed later
    distance = []
    voted_class = []
    count_1 = 0
    count_2 = 0
    count_3 = 0

    #If k is a list, run this function for each value for k
    if isinstance(k, list):
        for j in range(len(k)):
            if isinstance(i,list):
                for n in range(len(i)):
                    voted_class.append(kNN(point,k[j],i[n],closest_centroid,x,y))
            else:
                voted_class.append(kNN(point,k[j],i,closest_centroid,x,y))
    else:
        #Calculate distance (L^i norm) add this to a list with its index
        for j in range(len(x)):
            distance.append([j,((np.abs(x[j]-point[0])**i) + (np.abs(y[j]-point[0])**i))**(1/i)])

        #Sort said list by distance (shortsest first)
        #And find all interesting neighbors
        distance_sorted = sorted(distance,key=lambda distance:distance[1])
        neighbors = distance_sorted[:k]

        #Find the index for these neighbords
        neighbors_index = [i[0] for i in neighbors]

        #Count the class for these neighbors
        for n in neighbors_index:
            if closest_centroid[n] == 1:
                count_1 += 1
            elif closest_centroid[n] == 2:
                count_2 += 1
            else:
                count_3 += 1
        
        #See which class has a majority, return this class
        voted_class = np.argmax([count_1,count_2,count_3]) + 1
        print(f"With k = {k} and i = {i} the class of the new point is {voted_class}")
    return voted_class

def plotting(x: np.array,y: np.array,colour: str,label: str):
    """ Function that plots the data it recieves

        :param x: numpy array, array of x-coordinates for each data point
        :param y: numpy array, array of y-coordinates for each data point
        :param colour: string, string whished to be passed to the plotting ex 'r' for red and so on
        :param name: string, name for the file when saved
    """
    plt.scatter(x,y, color = colour, label = label )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Plot of data and clustering")
    plt.legend(loc = "best", fontsize = 8)



def main():
    """Main function of the script. Sends the file to be read and then
    sends the data to be plotted
    """
    
    filename = 'data_kmeans.txt'
    k = 3
    guess_x = [0,0,0]
    guess_y = [0,1,2]

    x,y = read_file('Assignment2/2.2/' + filename)

    centroids, closest_centroid= lloydsAlgo(k,guess_x,guess_y,x,y)
    print(f"The final position of C_1 is ({centroids[0][0][0]}, {centroids[1][0][0]})")
    print(f"The final position of C_1 is ({centroids[0][0][1]}, {centroids[1][0][1]})")
    print(f"The final position of C_1 is ({centroids[0][0][2]}, {centroids[1][0][2]})")

    point = [0,0]
    k = [3,7,11]
    i = [1,2,3]

    cluster = kNN(point, k, i, closest_centroid, x,y)


    for i in range(len(cluster)):
        plt.figure()
        plotting(centroids[0,:],centroids[1,:],colour = 'r',label = "Centroid final position")
        
        plotting(x[np.where(closest_centroid == 1)],y[np.where(closest_centroid == 1)],"m",label = "Cluster 1")
        plotting(x[np.where(closest_centroid == 2)],y[np.where(closest_centroid == 2)],"y",label = "Cluster 2")
        plotting(x[np.where(closest_centroid == 3)],y[np.where(closest_centroid == 3)],"c",label = "Cluster 3")
        
        if cluster[i] == 1:
            plotting(point[0],point[1],'m',label = "New point")
        elif cluster[i] == 2:
            plotting(point[0],point[1],'y',label = "New point")  
        else:
            plotting(point[0],point[1],'c',label = "New point")

        plt.savefig("Assignment2/2.2/plots/new_point_clustering" + str(i+1))     

    plt.figure()
    plotting(centroids[0,:],centroids[1,:],colour = 'r',label = "Centroid final position")
        
    plotting(x[np.where(closest_centroid == 1)],y[np.where(closest_centroid == 1)],"m",label = "Cluster 1")
    plotting(x[np.where(closest_centroid == 2)],y[np.where(closest_centroid == 2)],"y",label = "Cluster 2")
    plotting(x[np.where(closest_centroid == 3)],y[np.where(closest_centroid == 3)],"c",label = "Cluster 3")

    plotting(point[0],point[1],'k',label = "New point")
    plt.savefig("Assignment2/2.2/plots/new_point") 
    

    plt.figure()
    plotting(centroids[0,:],centroids[1,:],colour = 'r',label = "Centroid final position")
        
    plotting(x[np.where(closest_centroid == 1)],y[np.where(closest_centroid == 1)],"m",label = "Cluster 1")
    plotting(x[np.where(closest_centroid == 2)],y[np.where(closest_centroid == 2)],"y",label = "Cluster 2")
    plotting(x[np.where(closest_centroid == 3)],y[np.where(closest_centroid == 3)],"c",label = "Cluster 3")
    plt.savefig("Assignment2/2.2/plots/clustering") 

    plt.show()

if __name__ == '__main__':
    main()
