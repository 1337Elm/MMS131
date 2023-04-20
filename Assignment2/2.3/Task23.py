""" File that completes task 2.3 for assignment 2 in the course: MMS131

    Author: Benjamin Elm Jonsson 20011117 (2023) benjamin.elmjonsson@gmail.com
"""

#--------Imports------------

import numpy as np
import random
import matplotlib
import warnings

#---------------------------
#Ignore deprecation warnings
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


def read_file(filename: str) -> np.array:
    """Function that reads textfile and generates 2 lists and returning them

        :param filename: str, name of the file to be read
        :parm return: np.array, 2 np.arrays, 1 containing [x,y] and 1 for [g]
    """
    #Generate data from the textfile and put it in appropriate lists
    data = np.genfromtxt(filename)
    x = data[:,0]
    y = data[:,1]
    g = data[:,2]


    return np.array(x), np.array(y), np.array(g)


def initPop(K: int, num_param: int, param_range: int) -> np.array:
    """Function that initializes a population of given size, with given number of parameters

        :param K: int, population size
        :param num_param: int, number of parameters for each individual
        :param param_range: int, limit for each parameter (assuming its symmetrical ie range = [-lim,lim])
        :param return: np.array, 2d array of each individuals parameters
    """
    #Needed since random.uniform() uses [-param_range,param_range) instead of [-param_range,param_range]
    delta = 1e-6
    population = np.zeros((num_param,K))

    #Loop through each individual and each parameter and extract a random value for it
    for i in range(K):
        for j in range(num_param):
            population[j,i] = random.uniform(-param_range,param_range + delta)
    
    return np.array(population)


def evalPop(x: np.array,y: np.array, g: np.array, K: int,pop: np.array) -> int:
    """Function that evaluates the population and returns the fitness of it

        :param x: np.array, list of x values
        :param y: np.array, list of y values
        :param g: np.array, list of function values
        :param K: int, number of individuals in the population
        :param pop: np.array, 2d array of each individuals genome
        :param return: np.array, list of fitness values for each individual
    """
    #Initialize lists
    eps = np.zeros(K)
    fitness = np.zeros(K)

    #Loop through each individual and sum the square of the difference between g_hat and g
    for i in range(K):
        for j in range(K):
            g_hat = (1 + x[j]*pop[0,i] + x[j]**2*pop[1,i] + x[j]**3*pop[2,i])\
                /(1 + y[j]*pop[3,i] + y[j]**2*pop[4,i] + y[j]**3*pop[5,i])
            
            eps[i] += (g_hat-g[j])**2

    #Calculate eps according to given formula
    eps = np.sqrt(eps*(1/K))
    
    #Calculate fitness and return
    for i in range(len(eps)):
        fitness[i] = np.exp(-eps[i])

    return np.array(fitness)


def selectInd(fitness: np.array, p_tour: float, K: int) -> np.array:
    """Function that generates a list of the selected individuals

        :param fitness: np.array, list of fitness value for each individual
        :param p_tour: int, limit for selection of which individual to choose (used in order to be biased, so the
        individual with higher fitness often is selected)
        :param K: int, number of individuals

        :param return: np.array, list of selected individuals
    """
    selected_ind = []

    for j in range(K):
        for i in range(2):
            r = random.uniform(0,1)
            inds = [random.randint(0,K-1), random.randint(0,K-1)]
            if r < p_tour:
                if fitness[inds[0]] > fitness[inds[1]]:
                    selected_ind.append(inds[0])
                else:
                    selected_ind.append(inds[1])
            else:
                if fitness[inds[0]] < fitness[inds[1]]:
                    selected_ind.append(inds[0])
                else:
                    selected_ind.append(inds[1])

    return np.array(selected_ind) 



def crossOver(selected_ind: np.array, p_cross: float,K: int,population: np.array,cross_point: int,num_param: int,fitness: np.array) -> np.array:
    """ Function that performs the crossover of genomes for each pair of individual
        in order for the population to evolve.

        :param selected_ind: np.array, list of individuals that make it to the next gen
        :param p_cross: int, probability of a cross over of genome
        :param K:, int, number of individuals
        :param population: np.array, 2d array of each individual and their genome

        :return new_gen: np.array, 2d array of the new generation and their genomes
    """
    #Initiate new 2d array for the population
    new_pop = np.zeros((num_param, K))

    #Loop through every other individual check if it is supossed to be crossed
    #If so swap parameters up til the crossing point with the next individual
    #Add the modified individuals to the new list and return
    for i in range(K-1):
        r = random.uniform(0,1)
        if r < p_cross:
            temp = population[:cross_point,selected_ind[i]]
            population[:cross_point,selected_ind[i]] = population[:cross_point,selected_ind[i+1]]
            population[:cross_point,selected_ind[i+1]] = temp

            new_pop[:,i] = population[:,selected_ind[i]]
            new_pop[:,i+1] = population[:,selected_ind[i+1]]
            i += 2
    
    #Elitism
    new_pop[:,0] = population[:,np.argmax(fitness)]
    return np.array(new_pop)
        

def mutation(new_gen: np.array,p_mut: float, p_creep: float,creepRate: float,K: int,num_param: int,param_range: int) -> np.array:
    """ Function that mutates the population

        :param new_gen: np.array, 2d list of the population after cross over
        :param p_mut: float, probability of a mutation
        :param p_creep: float, probability of creep mutation
        :param creepRate: float, how much the genome is supossed to creep 
        :param mutated_pop: np.array, returns an updated list after the population has mutated
    """
    #Initiate new list for mutated population
    mutated_pop = np.zeros((num_param,K))

    delta = 1e-6

    #Loop through all individuals genomes and check if its supossed to be mutated
    #If so check if mutation is supossed to be creep or ordinary
    #Add mutated individual to new list and return
    for i in range(K):
        for j in range(num_param):
            r = random.uniform(0,1)
            if r < p_mut:
                if r < p_creep:
                    new_gen[j,i] = new_gen[j,i] + random.uniform(-creepRate/2,creepRate/2)
                    mutated_pop[j,i] = new_gen[j,i]
                else:
                    new_gen[j,i] = random.uniform(-param_range,param_range + delta)
                    mutated_pop[j,i] = new_gen[j,i]
            else:
                mutated_pop[j,i] = new_gen[j,i]

    return np.array(mutated_pop)    


def main():
    """Main function of the script. Iterating through the genetic algorithm procedure
    """
    #Hard code specific case
    filename = 'data_ga.txt'
    K = 100
    num_param = 6
    param_range = 2
    p_tour = 0.75
    p_cross = 0.85
    cross_point = 2
    p_mut = 0.125
    p_creep = 0.5
    creep_rate = 0.01

    x, y, g = read_file('Assignment2/2.3/' + filename)

    #Go step by step through the metholodgy 
    #init population, evaluate, cross-over, mutation, iterate
    pop = initPop(K,num_param,param_range)

    fitness = evalPop(x,y,g,K,pop)
    
    selected_ind = selectInd(fitness,p_cross,K)

    new_gen = crossOver(selected_ind,p_cross,K,pop, cross_point,num_param,fitness)
    mutated_gen = mutation(new_gen,p_mut,p_creep,creep_rate,K,num_param,param_range)


if __name__ == '__main__':
    main()
