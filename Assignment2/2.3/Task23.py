""" File that completes task 2.3 for assignment 2 in the course: MMS131

    Author: Benjamin Elm Jonsson 20011117 (2023) benjamin.elmjonsson@gmail.com
"""

#--------Imports------------

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import warnings
import time
from tqdm.auto import tqdm

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
    """Function that calculates the estimate g_hat, then calculates epsilon aswell as the fitness.

        :param ind_gene: list, a list of the genes for the current individual
        :param x: np.array, array of x values from txt file
        :param y: np.array, array of y values from txt file
        :param g: np.array, correct values from txt file
        :param K: int, population size

        :param return: int, fitess for the current individual
    """
    g_hat = (1 + x*ind_gene[0] + x**2*ind_gene[1] + x**3*ind_gene[2])\
        /(1 + y*ind_gene[3] + y**2*ind_gene[4] + y**3*ind_gene[5])
        
    eps = np.sqrt(np.sum((g_hat-g)**2)*(1/K))
    return np.exp(-eps)


def selectInd(fitness: np.array, p_tour: float,K: int) -> int:
    """Function that takes 2 random individuals of the population and selects 1 of them based
        their fitness. The individual with high fitness is more likely to be chosen.

        :param fitness: np.array, fitness for each individual
        :param p_tour: float, probability that individual with higher fitness is chosen
        :param K: int, populationsize
        
        :param return: int, index for the chosen individual
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
    """Function that crosses over two genes (all chromosomes up to a random point)

        :param gene_1: list, list of chromosomes for individual 1
        :param gene_2: list, list of chromosomes for individual 2
        :param num_param: int, number of chromosomes per individual

        :param return: np.array, array of the two new individuals
    """
    new_inds = []
    new_gene1 = []
    new_gene2 = []

    cross_point = random.randint(0,num_param)
    for i in range(len(gene_1)):
        if i <= cross_point:
            new_gene1.append(gene_2[i])
            new_gene2.append(gene_1[i])
        else:
            new_gene1.append(gene_1[i])
            new_gene2.append(gene_2[i])
    
    new_inds.append(new_gene1)
    new_inds.append(new_gene2)

    return np.array(new_inds)
        

def mutation(gene: np.array,p_mut: float, p_creep: float,creepRate: float,param_range: int) -> np.array:
    """Function that mutates each chromose based on a probability. If the chromosome is supossed to be mutated.
        It is then decided based on probability if the mutation is supossed to be creep-mutation of full mutation.

        :param gene: np.array, list of chromosomes for the current individual
        :param p_mut: float, probability that a chromosome is to be mutated
        :param p_creep: float, probability that the mutation is a creep-mutation
        :param param_range: int, the allowed range for each parameter

        :param return: np.array, new list of chromosomes for the current individual
    """
    new_gene = np.zeros(len(gene))
    delta = 1e-6

    for i in range(len(gene)):
        r = random.uniform(0,1)
        if r < p_mut:
            r = random.uniform(0,1)
            if r < p_creep:
                new_gene[i] = gene[i] + random.uniform(-creepRate/2,creepRate/2 + delta)
            else:
                new_gene[i] = random.uniform(-param_range,param_range + delta)
        else:
            new_gene[i] = gene[i]
    return new_gene 


def runAlgo(param_range: int, p_tour: int, p_cross: int, p_mut: int, p_creep: int, creep_rate: int,plot: bool) -> list:
    
    start = time.time()

    filename = 'data_ga.txt'
    x, y, g = read_file('Assignment2/2.3/' + filename)
    K = 100
    num_param = 6

    num_gens = 10000
    best_fitness_gen = np.zeros(num_gens)
    mean_fitness = np.zeros(num_gens)
    fitness = np.zeros(K)
    max_fitness = 0

    #Go step by step through the metholodgy 
    #init population, evaluate, cross-over, mutation, iterate
    pop = initPop(K,num_param,param_range)

    for i in range(num_gens):
        max_fitness_gen = 0
        best_ind_gen = None
        for j in range(K):
            fitness[j] = evalInd(pop[:,j],x,y,g,K)
            if fitness[j] > max_fitness_gen:
                max_fitness_gen = fitness[j]
                best_ind_gen = pop[:,j]
            if fitness[j] > max_fitness:
                max_fitness = fitness[j]
                best_Ind = j

        best_fitness_gen[i] = evalInd(best_ind_gen,x,y,g,K)
        mean_fitness[i] = np.mean(fitness)
        if i < num_gens:
            temp_pop = pop
            temp_pop[:,0] = best_ind_gen
            for j in range(0,K,2):
                if j != 0:
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
            
            for n in range(len(temp_pop[1,:])):
                if n != 0:
                    mutatedGene = mutation(temp_pop[:,n],p_mut,p_creep,creep_rate,param_range)
                    temp_pop[:,n] = mutatedGene
            pop = temp_pop
    
    if plot == True:
        print(f"the most correct parameters are {pop[:,best_Ind]}")
        end = time.time()
        print(f"Time spent: {end-start}")

        fig = plt.figure()
        plt.plot(np.linspace(0,num_gens, num = num_gens),best_fitness_gen, 'b',label = "Best fitness/generation")
        plt.plot(np.linspace(0,num_gens,num_gens), mean_fitness,'k',label = "Mean fitness/generation")
        #plt.axis([0,num_gens,0,1])
        plt.xlabel("generation [-]")
        plt.ylabel("fitness [-]")
        plt.title("Plot of best fitness as a function of generation")
        plt.legend(loc = "best")
        #plt.savefig("Assignment2/2.3/plots/best_fitness.jpeg")
        plt.show()

    return max_fitness


def find_best_params():

    param_range = [2,1,3,4]
    p_tour = [0.2,0.4,0.75,0.9]
    p_cross = [0.2,0.4,0.75,0.9]
    p_mut = [0.125,0.4,0.6,0.8]
    p_creep = [0.2,0.5,0.7,0.9]
    creep_rate = [0.01,0.1,0.2,0.3]

    progress_bar = tqdm(total = len(param_range)**6,position=0,leave=True,colour="green")

    max_fitness = 0
    best_params = []
    for i1 in range(len(param_range)):
        for i2 in range(len(p_tour)):
            for i3 in range(len(p_cross)):
                for i4 in range(len(p_mut)):
                    for i5 in range(len(p_creep)):
                        for i6 in range(len(creep_rate)):
                            max_fitness_gen = runAlgo(param_range[i1],p_tour[i2],p_cross[i3],p_mut[i4],p_creep[i5],creep_rate[i6],plot = False)
                            if max_fitness_gen > max_fitness:
                                max_fitness = max_fitness_gen
                                best_params = [param_range[i1],p_tour[i2],p_cross[i3],p_mut[i4],p_creep[i5],creep_rate[i6]]
                            progress_bar.update(1)
    
    return best_params,max_fitness


def main():
    """Main function of the script. Calling the function that goes through the metholodgy
    """
    #Hard code specific case
    param_range = 2
    p_tour = 0.75
    p_cross = 0.75
    p_mut = 0.125
    p_creep = 0.5
    creep_rate = 0.01

    case = input("Do you want to find best parameters or run normal case? (F/N)")
    if case == "N":
        runAlgo(param_range,p_tour,p_cross,p_mut,p_creep,creep_rate,plot = True)
    elif case == "F":
        params, maxfitness = find_best_params()

        print(f"The best parameters are the following\n")
        print(params)
        print("——————————————————————————————————-")
        print(f"The best fitness with these parameters is {maxfitness}")

        runAlgo(params[0],params[1],params[2],params[3],params[4],params[5],plot = True)



if __name__ == '__main__':
    main()
