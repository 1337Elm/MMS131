""" File that completes task 2.3 for assignment 2 in the course: MMS131

    Author: Benjamin Elm Jonsson 20011117 (2023) benjamin.elmjonsson@gmail.com
"""

#--------Imports------------

import numpy as np
import random
import matplotlib.pyplot as plt
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


def evalInd(ind_gene: list,x: np.array,y: np.array, g: np.array, K: int) -> int:
    """Function that evaluates the population and returns the fitness of it

        :param x: np.array, list of x values
        :param y: np.array, list of y values
        :param g: np.array, list of function values
        :param K: int, number of individuals in the population
        :param pop: np.array, 2d array of each individuals genome
        :param return: np.array, list of fitness values for each individual
    """
    eps = 0
    #Loop through each individual and sum the square of the difference between g_hat and g
    for j in range(K):
        g_hat = (1 + x[j]*ind_gene[0] + x[j]**2*ind_gene[1] + x[j]**3*ind_gene[2])\
            /(1 + y[j]*ind_gene[3] + y[j]**2*ind_gene[4] + y[j]**3*ind_gene[5])
            
        eps += (g_hat-g[j])**2

    #Calculate eps according to given formula
    eps = np.sqrt(eps*(1/K))
    
    return np.exp(-eps)


def selectInd(fitness: np.array, p_tour: float,K: int) -> int:
    """Function that generates a list of the selected individuals

        :param fitness: np.array, list of fitness value for each individual
        :param p_tour: int, limit for selection of which individual to choose (used in order to be biased, so the
        individual with higher fitness often is selected)
        :param K: int, number of individuals

        :param return: int, index for selected individual
    """
    r = random.uniform(0,1)
    inds = [random.randint(0,K-1), random.randint(0,K-1)]
    if r < p_tour:
        if fitness[inds[0]] > fitness[inds[1]]:
            return inds[0]
        else:
            return inds[1]
    else:
        if fitness[inds[0]] < fitness[inds[1]]:
            return inds[0]
        else:
            return inds[1]


def crossOver(gene_1: list,gene_2: list,num_param: int) -> np.array:
    """ Function that performs the crossover of genomes for a pair of individuals
        in order for the population to evolve.

        :param gene_1: np.array, genome for individual one
        :param gene_2: np.array, genome for individual two
        :param num_param:, int, number of individuals

        :return new_gen: np.array, list of the two new individuals
    """
    #Loop through every other individual check if it is supossed to be crossed
    #If so swap parameters up til the crossing point with the next individual
    #Add the modified individuals to the new list and return
    new_inds = []

    cross_point = random.randint(0,num_param)
    temp = gene_1[:cross_point]
    gene_1[:cross_point] = gene_2[:cross_point]
    gene_2[:cross_point] = temp
    
    new_inds.append(gene_1)
    new_inds.append(gene_2)
    
    return np.array(new_inds)
        

def mutation(gene: np.array,p_mut: float, p_creep: float,creepRate: float,param_range: int) -> np.array:
    """ Function that mutates the population

        :param new_gen: np.array, 2d list of the population after cross over
        :param p_mut: float, probability of a mutation
        :param p_creep: float, probability of creep mutation
        :param creepRate: float, how much the genome is supossed to creep 
        :param mutated_pop: np.array, returns an updated list after the population has mutated
    """
    new_gene = np.zeros(len(gene))
    delta = 1e-6

    #Loop through all individuals genomes and check if its supossed to be mutated
    #If so check if mutation is supossed to be creep or ordinary
    #Add mutated individual to new list and return
    for i in range(len(gene)):
        r = random.uniform(0,1)
        if r < p_mut:
            if r < p_creep:
                new_gene[i] = gene[i] + random.uniform(-creepRate/2,creepRate/2)
            else:
                new_gene[i] = random.uniform(-param_range,param_range + delta)
        else:
            new_gene[i] = gene[i]

    return new_gene 


def main():
    """Main function of the script. Iterating through the genetic algorithm procedure
    """
    #Hard code specific case
    filename = 'data_ga.txt'
    K = 100
    num_param = 6
    param_range = 2
    p_tour = 0.75
    p_cross = 0.75
    p_mut = 0.125
    p_creep = 0.5
    creep_rate = 0.01

    x, y, g = read_file('Assignment2/2.3/' + filename)

    num_gens = 100
    best_fitness = np.zeros(num_gens)
    fitness = np.zeros(K)
    max_fitness = 0
    max_fitness_gen = 0

    #Go step by step through the metholodgy 
    #init population, evaluate, cross-over, mutation, iterate
    pop = initPop(K,num_param,param_range)

    for i in range(num_gens):
        print(pop[:,0])
        max_fitness_gen = 0
        for j in range(K):
            fitness[j] = evalInd(pop[:,j],x,y,g,K)
            if fitness[j] > max_fitness:
                max_fitness = fitness[j]
                best_Ind = j
            if fitness[j] > max_fitness_gen:
                max_fitness_gen = fitness[j]
                best_ind_gen = j

        best_fitness[i] = max_fitness_gen

        for j in range(K-1):
            temp_pop = pop

            ind_1 = selectInd(fitness,p_tour,K)
            ind_2 = selectInd(fitness,p_tour,K)

            r = random.uniform(0,1)
            if r < p_cross:
                new_inds = crossOver(pop[:,ind_1],pop[:,ind_2],num_param)

                temp_pop[:,j] = new_inds[0]
                temp_pop[:,j+1] = new_inds[1]
            else:
                temp_pop[:,j] = pop[:,ind_1]
                temp_pop[:,j+1] = pop[:,ind_2]
            
            temp_pop[:,0] = pop[:,best_ind_gen]
            j += 2

        for n in range(1,K):
            mutatedGene = mutation(pop[:,n],p_mut,p_creep,creep_rate,param_range)
            temp_pop[:,n] = mutatedGene
        pop = temp_pop
        print(pop[:,0])
    
    #print(pop[:,best_Ind])

        

    fig = plt.figure()
    plt.plot(np.linspace(0,num_gens, num = num_gens),best_fitness, 'b')
    #plt.axis([0,num_gens,0,1])
    plt.xlabel("generation [-]")
    plt.ylabel("fitness")
    plt.show()


if __name__ == '__main__':
    main()
