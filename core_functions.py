import numpy as np
from random import random
import random as rand
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy.sparse import csr_matrix
from scipy.integrate import solve_ivp
from scipy import io 

import copy
import time
from time import sleep
import pickle
import os

Olive = '#CAFF70'
Blue = '#2CBDFE'
Sea = '#1E90FF'
Green = '#47DBCD'
Baby_Pink = '#F3A0F2'
Magenta = '#EE1289'
Purple = '#BF3EFF'
Violet = '#7D26CD'
Lilac = '#8470FF'
Dark_blue = '#0000CD'
Green_tea = '#66CDAA'
Pink = '#FF69B4'
Hot_pink = '#EE00EE'
Barbie = '#EE30A7'
Dream = '#00EE76'
Turquoise = '#00E5EE'
Berry = '#D02090'
Sky = '#87CEFF'

updated_colour = [Blue, Baby_Pink, Green,Sea, Olive, Purple, Violet,Dark_blue,Magenta,Lilac,Green_tea,Pink,
                 Hot_pink,Barbie,Dream,Turquoise,Berry,Sky]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color = updated_colour)





def mutation_freq(mut_prob, n, size):
    '''
    A function which gives the number of mutants in a sub-population based on binomial distribution
    Input: 
        mut_prob - probability of success of a particular event occuring, here success is mutation 
        n - the size of sub-population, or number of cells in each subpopulation. The number of independent trial
            Make sure the selected 'n' are more than 0
        size - the number of repeats of an event. For mutation, choose size = len(n) going over once
               For example, size = 10 means flipping a coin 10 times to get number of successes of 'heads'      
    Output: 
        The number of success for a given number of trials, the number of mutants in a sub-population (1d array)     
    '''
    
    num_mutants = np.random.binomial(n = n, p = mut_prob,size=size) # binomial distribution of subpopulation    
    return num_mutants

def mutation_distribution():              # AY
    return np.round(np.random.normal(),2)


def mutation_distribution2(mutants,s_pos,s_neg):

    u2 = np.random.random(mutants)
    lim = s_neg/(s_pos + s_neg)
    m = np.zeros(mutants)
    
    index = np.where([u2[i]<=lim for i in range(mutants)])[0]
    m[index] = [s_neg*np.log(u2[index[i]]*(s_pos+s_neg)/s_neg) for i in range(len(index))]

    index = np.where([u2[i]>lim for i in range((mutants))])[0]
    m[index] = [-s_pos*np.log((1-u2[index[i]])*(s_pos+s_neg)/s_pos) for i in range(len(index))]
    
    m = np.array([-1 if m[i]<-1 else m[i] for i in range(len(m))])
    return m 


def rand_mut_param(min_param, max_param, mutation_distribution, mutants):
    
    '''
    Selects a random parameter and its corresponding random mutation
    Input:
        min_param - minimum parameter value, marker 
        max_param - maximum parameter value, marker
        mutation_distribution - type of distribution used to select mutation change
    Output:
        Two values, first is the randomly selected parameter (integer), second is the mutation for parameter (float)
        (int,float)
    '''
    min_param=0
    select_parameter = rand.randrange(min_param, max_param)
    mutation_change = mutation_distribution() # mutation range from the 'change' column
    
    return select_parameter, mutation_change

def mutation_distribution_n(no_mutants):              # AY
    return np.round(np.random.normal(size=no_mutants),2)


def mut_param_np(max_param, mutation_distribution,no_mutants):
    
    min_param = 0
    
    select_parameter = np.random.uniform(min_param,max_param,no_mutants)
    mutation_change = mutation_distribution_n(no_mutants)
    
    return select_parameter, mutation_change

def ancestor_min(table,pointer):
    
    '''
    A function which takes away 1 cell from selected ancestral subpopulation
    Input:
        table - the table with the ancestor population information
        pointer - the location of row of ancestor in the table, the index of ancestor row
    Output:
        Updated table with 1 cell taken away from ancestor 
    '''
    table0 = copy.deepcopy(table)
    table0.at[pointer,'Number'] = table0.at[pointer,'Number'] - 1
    return table0

def add_subpop(table, pointer, select_parameter, mutation_change):
    '''
    A function which generates a new sub-population with 1 cell resulting from mutation and its closeout line  
    Input: 
        table - the original table with all the ancestral rows
        pointer - the location of ancestor sub-population. The row index of ancestor sub-population where mutation occured
        select_parameter - random selected parameter 
        mutation_change - random change due to mutation 
    
    Output: Table with two rows, where 1st row is the new sub-population with 1 cell and its closeout line
            Inserts below the ancestor, followed by the remaining subpopulation
    '''
    new_subpop = pd.DataFrame({'Parameter':select_parameter,'Change':mutation_change,
                 'Number': 1,'Length':table['Length'][pointer]}, index = [0])        
    closeout = pd.DataFrame({'Parameter': select_parameter, 'Change': - mutation_change,
                         'Number': 0,'Length':0}, index = [0]) 
    table1 = pd.concat([new_subpop,closeout]).reset_index(drop = True)
    table = pd.concat([table.iloc[:pointer+1],table1,table.iloc[pointer+1:]]).reset_index(drop = True)
    
    return table


def mutation(table,min_param,max_param,mutation_p):
    
    '''
  A function which adds mutation to an ancestor 
  Input: 
      table - the starting table with ancestor, with columns: parameter, change and number of cells  
      min_param - the 1s parameter index
      max_param - the last parameter index, total number of parameters
      mutation_p - mutation probability
  Return:
      A table with added mutations to random parameters. Branching from ancestor to different subpopulation
      
    '''
    s_neg = 0.067
    s_pos = 0.05
    num_mutation = mutation_freq(mutation_p,table['Number'],len(table))    
    update_index = np.arange(len(table))
    for i in range(len(num_mutation)):
        if num_mutation[i] != 0:
            for each in range(num_mutation[i]):
                parameter, change = rand_mut_param(min_param,max_param,mutation_distribution,1)
                if change != 0:
                    ancestor_min(table, update_index[i])
                    table = add_subpop(table,update_index[i],parameter,change)
                    update_index[i+1:] +=2          
    else:
        return table;


### Growth cell updated ###
def growth_(s, table):
    '''
  This function implements growth on subpopulation depending on updated biomass after dynamics, chemical mediated interaction.
  If the size = 2, the number of cells in sub-population will multiply by 2 and new size/2, else the length and number will remain. 
  The new updated sub-population will have new size = size/2
  Input:
         s - the updated biomass after the dynamics calculations, i.e solving coupled ODE's. Type: 1d array.
         table - table after dynamics. Type: pandas table [Parameter,Change,Number,Length].
  Return:
        The updated table with size ('Length') and the number ('Number') of cells for each subpopulation in the table.         
    '''
    number_cell = table.loc[:,'Number']
    length_cell = table.loc[:,'Length']
    biomass = np.reshape(s,(len(number_cell),1))
    new_number = []
    new_size = []   
    for row in range(len(table)): 
        if number_cell[row] == 0:
            new_number.append(number_cell[row]) # else the number of cells remain the same
            new_size.append(length_cell[row]) # old size remains if subpopulation i.e row is extinct (Number=0). To distinguish opener.
        else:
            size = biomass[row][0]/number_cell[row]
            if size >= 2:
                new_number.append(number_cell[row]*2)      
                size = size/2 
                new_size.append(size)
            else:
                new_number.append(number_cell[row]) # else the number of cells remain the same
                new_size.append(size) # old size remains if less than 2
            
    new_number = np.reshape(new_number,(len(new_number),1)) # updated number of cells in a column
    new_size = np.reshape(new_size,(len(new_size),1)) # updated size of each cell after growth
    
    table1 = copy.deepcopy(table)
    table1.loc[:,'Number'] = new_number
    table1.loc[:,'Length'] = new_size
    table_final = table1.fillna(0)
    
    return table_final


### Growth cell updated ###
def growth_number(s, table):
    '''
  This function implements growth on subpopulation depending on updated biomass after dynamics, chemical mediated interaction.
  If the size = 2, the number of cells in sub-population will multiply by 2 and new size/2, else the length and number will remain. 
  The new updated sub-population will have new size = size/2
  Input:
         s - the updated biomass after the dynamics calculations, i.e solving coupled ODE's. Type: 1d array.
         table - table after dynamics. Type: pandas table [Parameter,Change,Number,Length].
  Return:
        The updated table with size ('Length') and the number ('Number') of cells for each subpopulation in the table.         
    '''
    number_cell = table.loc[:,'Number']
    biomass = np.reshape(s,(len(number_cell),1))
    new_number = []
    new_size = []   
    for row in range(len(table)): 
        if number_cell[row] == 0:
            new_number.append(s[row]) # else the number of cells remain the same
        else:
            new_number.append(s[row]) # else the number of cells remain the same

    new_number = np.reshape(new_number,(len(new_number),1)) # updated number of cells in a column
    
    table1 = copy.deepcopy(table)
    table1.loc[:,'Number'] = new_number
    table_final = table1.fillna(0)
    
    return table_final

### Sparse and dense matrix ###
def initial_condition(max_param):
    
    '''
    Determines initial condition values for each parameter, i.e ancestral values for each parameter
    Input:
        max_param - maximum number of parameter, total parameters
    Output:
        An array with initial conditions which gives ancestral parameter values at t=0
    '''    
    initial_condition_params = random.uniform(size = max_param)
    
    return initial_condition_params  

def sparse(table,min_param,max_param,initial_condition):
    
    '''
    Function which converts table into the sparse csr_matrix form
    Input:
        table - table to be stored as sparse csr_matrix form
        min_param - the first parameter index
        max_param - the maximum parameter value, the total number of parameters
        initial - initial conditions with initial values for all parameters, max_param number (1d array)
    Output:
        A csr_matrix form which stores the content of the table
        
    '''
    #breakpoint()
    
    col = np.concatenate([np.arange(min_param,max_param),table['Parameter']]) # columns are the parameter values
    row = np.concatenate([np.int64(np.zeros(max_param)), np.arange(len(table))]) # index of the rows in the table
    data = np.concatenate([initial_condition, table['Change']]) # the changes from the table usually after mutation
    
    return csr_matrix((data,(row,col)))

def number_cells(table):
    
    '''
    Gets the number of cells from table 
    Input:
        table - the table which contains the numbers column 
    Output: 
        Column which has all the number of cells in each subpopulation
    '''
    
    number = np.zeros((len(table),1))
    for i in range(len(table)):
        number[i] = table['Number'][i]
    return number

def dense(sparse):
    
    '''
    Takes in a csr_matrix format to get the dense form
    Input: 
        dense - A sparse csr_matrix format which contains initial parameter values and changes from mutation 

    Output: 
        The total dense matrix from sparse matrix 
    '''
    return np.cumsum(sparse.toarray(), axis=0)

def sparse_dense(table,max_param,ancestor_row):
    '''
    Combining the sparse to dense operation as one step
    Input: 
        table - the table after the mutation with updated population sizes and subpopulations
        max_param - the total number of parameters, based on number of chemicals, 
                    number of columns of the matrix with all the parameters K, r0, alpha, beta, rho_plus, rho_minus
        ancestor_row - initial value for each parameter of the ancestral population
                       single row with the total number of parameters column 
    Output: 
        The dense matrix with updated population sizes after mutation 
    
    '''
# create sparse matrix from the table with (ancestor + 1 round of mutation)
    Z = sparse(table, 0, max_param,ancestor_row) # creating a sparse form from the resulting mutation 
    X = dense(Z) # total dense matrix for species 1 after 1 round of mutation
    return X
### End of sparse matrix ###

def biomass(table):
    '''
    The biomass of species, which is the number of cells * cell length
    Input: 
        table - table which contains the species and subpopulations and their population size and length of cells
    Output: 
        The biomass value for a species and its subpopulation 
    
    '''
    table = table.fillna(0)
    biomass = table['Number'] * table['Length'] 
    return biomass

def label_maker(biomass):
    '''
    Determine how many subpopulation there are without closeout lines
    Input: 
        biomass - a column which contains the biomass of species, biomass = number cells * length
    Output:
        The number, size of the non-zero biomass from a mutation
        Useful in making the labels of the plots 
    '''
    biomass_label =[]
    for i in range(len(biomass)):
        if biomass[i] != 0:
            biomass_label.append(biomass[i])
        else:
            pass
    return len(biomass_label)

### Repacking the dense form to seperate parameters and back ###

def get_ncols(params, param_names): # run once 
    
    '''
    To get the number of columns for each parameter in a dictionary
    Input: 
        params - the parameter dictionary, including the array of values for each component
        param_names - the name of the parameters
    Return: 
        Gets the number of columns for each parameter
    '''
    return np.array([params[_].shape[1] for _ in param_names]);

def params_dict_to_matrix(params, param_names): # run once to establish the table

    '''
      Inputs all the arrays in a dictionary and input it as a single large array ordered one after another
      In this case, put all parameters K, r0, alpha, beta, rho_plus and rho_minus into one matrix with columns
      equal to the size of all the columns of individual parameters

      Input: 
        params - the parameter names in a dictionary, and the input for each element in the dictionary
                 these are the parameter values which can be of form n_s x n_c or vice versa
        param_names - the names of all the parameters in the dictionary
      Output: 
        A single large array containing the same number of columns as the sum of individual parameter columns

    '''
    result = []
    for spec in range(params["num_spec"]):
        spec_params = []
        for param_name in param_names:
            spec_params.append(params[param_name][spec,:])
        result.append(np.concatenate(spec_params))
    return np.array(result);

def params_matrix_to_dict(params_as_matrix, param_names, ncols): # run each time you need dynamics
    '''
      Splits the large matrix or n-dimensional array into their original specific parameter matricies

      Input: 
        params_as_matrix - a single large matrix or n-dimensional array containing all the parameters
        param_names - the names of all the parameters in the dictionary
        n_cols - the number of columns for each individual parameter
      Output: 
        A dictionary which has all the parameter values in their respective small matricie 

    '''
    params = {}
    cutpoints = np.zeros(ncols.size+1).astype(int) 
    cutpoints[1:] = np.cumsum(ncols)
    for i in range(len(param_names)):
        params["num_spec"] = params_as_matrix.shape[0]
        begin_col = cutpoints[i]
        finish_col = cutpoints[i+1]
        param_name = param_names[i]
        param = params_as_matrix[:, begin_col : finish_col]
        params[param_name] = param
    
    return params;


def cut_species(s,table):
    '''
    A function which gives cut points for seperate species and their subpopulation
    Input:
        s - the updated species biomass from dynamics calculation
        table - the updated table after mutation with species and their subpopulation
    Return:
        All species biomass cut at the correct points seperating different species
    '''
    s = np.array(s).flatten()
    cuts = [table[i].shape[0] for i in range(len(table))]
    s_all = []
    upd = np.arange(len(cuts))
    for n in range(len(cuts)):
        m = s[upd[n]:upd[n]+cuts[n]]
        upd[n+1:]+=cuts[n]-1
        s_all.append(m)
    return s_all

### Dynamics run 1 ###
def dSCdt(SC, num_spec, r0, K, alpha, beta, rho_plus, rho_minus):
    """
    Parameters:

    SC (array): an array of species and chemical abundances in which species
        are listed before chemicals
    num_spec (int): number of species
    r0 (2d numpy.array): num_spec x 1 array of intrinsic growth rates
    K (2d numpy.array): num_spec x num_chem array of K values
    alpha (2d numpy.array): num_chem x num_spec array of consumption constants
    beta (2d numpy.array): num_chem x num_spec array of production constants
    rho_plus (2d numpy.array): num_spec x num_chem array of positive influences
    rho_minus (2d numpy.array): num_spec x num_chem array of negative influences

    Returns:
    An array of rates of change in which species are listed before chemicals
    """
    S = np.reshape(SC[:num_spec], [num_spec,1])
    C = np.reshape(SC[num_spec:], [len(SC) - num_spec, 1])
    # compute K_star
    K_star = K + C.T
    # compute K_dd
    K_dd = rho_plus * np.reciprocal(K_star)
    min_rho = np.reciprocal(K) * rho_minus
    # compute lambda
    Lambda = np.matmul(K_dd - min_rho, C)
    # compute dS/dt
    S_prime = (r0 + Lambda) * S
    # compute K_dag
    C_broadcasted = np.zeros_like(K.T) + C
    K_dag = np.reciprocal(C_broadcasted + K.T) * C_broadcasted
    # compute dC/dt
    C_prime = np.matmul(beta - (alpha * K_dag), S)
    SC_prime = np.vstack((S_prime, C_prime))
    return SC_prime

def sc_prime_std(t, SC, parameters):
    """
    defines the derivative in a format more friendly to scipy.integrate.ode
    """
    result = dSCdt(SC,
                 parameters["num_spec"],
                 parameters["r0"],
                 parameters["K"],
                 parameters["alpha"],
                 parameters["beta"],
                 parameters["rho_plus"],
                 parameters["rho_minus"])
    return np.reshape(result, result.size).tolist()

### Dynamics run1 end ###

def split_rho(rho):
    """
    Parameters:
    rho (2d numpy.array): rho matrix

    Returns: (rho_plus, rho_minus)
    rho_plus (2d numpy.array): a matrix whose nonzero elements are the positive
        elements of rho
    rho_minus (2d numpy.array): a matrix whose nonzero elements are the negative
        elements of rho
    """
    rho_plus = rho *(rho>=0)
    rho_minus = rho * (rho < 0)
    return (rho_plus, rho_minus)

def dynamics(dydt,y0,t):

        '''
    A function which solves ODE for a timespan 't' and for initial conditions 'y0'. 
    Particularly, for using exact integration method to solve for all dynamical variables i.e all biomass and chemicals
    Input: 
        dydt - the function to be solved by numerical integrator, which has only two variables 't' and 'y0'
        y0 - initial conditions for t=0 to start solving the ODE from somewhere
        t - the duration or timespan to solve the ODE i.e integration range: {t0,t} t0-start time, t-final time
    Output:
        The array of updated dynamical values for species density 'S' and chemical concentration 'C'
        '''

        y = solve_ivp(fun=dydt,
                        t_span=[0, t], y0=y0)  
        return y

def cell_cycle_list(params1,c0,table,t,mutation_p,ExtTh):
    '''
    The updated cell cycle function. Everything in list form
    '''
    param_names = [i for i in params1 if i != 'num_spec'] # list of all names of the parameters in dictionary
    ncols1 = get_ncols(params1,param_names) # getting the number of columns for each parameter, K, rho etc.
    params_as_matrix = params_dict_to_matrix(params1,param_names) # combining all the columns into one big matrix
    n_c = params1['K'].shape[1]
    n_s = params1['num_spec']

    max_param = params_as_matrix.shape[1] # numer of columns, all parameters i.e dynamical quantities K, alpha, r0...
    mutation0 = [mutation(table[i],0,max_param,mutation_p) for i in range(len(table))]

    X = np.concatenate([sparse_dense(mutation0[i],max_param,params_as_matrix[i]) for i in range(len(mutation0))])

    biomass0 = np.concatenate([biomass(mutation0[i]) for i in range(len(mutation0))])
    dynamic_tot0 = params_matrix_to_dict(X, param_names, ncols1)

    def dydt(t, sc): 
        '''
    Changing format to be solved by numerical integration. The format needs the ODE to have 2 variables
    Input:
        t - time span to solve the ODE
        sc - initial condition values for all the species and chemicals (array of length n_s + n_c)
    Output: 
        The correct equation format to be solved in solve_ivp, where input is time and initial condition
        '''
        return sc_prime_std(t, sc, dynamic_tot0)

    s_label = ['s{}'.format(i+1) for i in range(label_maker(biomass0))]
    c_label = ['c{}'.format(i+1) for i in range(n_c)]

    y0 = np.concatenate((biomass0, c0)) # initial conditions for dynamics calculations

    # sovle the ODE to get dynamics 
    y = dynamics(dydt,y0,t)

    ## plot the results to see ODE in action for each timestep 
    # plt.figure()
    # plt.semilogy(y.y.T[:,:len(biomass0)], '.-',alpha=0.5)
    # plt.xlabel('Timestep')
    # plt.ylabel('Change in biomass')
    # plt.legend(s_label, loc = 'best')
    # plt.show()

    # plt.figure()
    # plt.semilogy(y.y.T[:,-n_c:],'.-',alpha=0.5)
    # plt.xlabel('Timestep')
    # plt.ylabel('Change in chemical concentration')
    # plt.legend(c_label, loc = 'best')
    # plt.show()

    nCell = y.y.T[-1:,:-n_c]
    nCell = np.array([0 if i<ExtTh else i for i in nCell])
    chemicals = y.y.T[-1:,-n_c:][0]

    s_cut = cut_species(nCell,mutation0)
    updated_table = [growth_(s_cut[i],mutation0[i]) for i in range(len(mutation0))]

    return nCell, chemicals, updated_table


def maturation(params1,c0,table0,t,step,mutation_p,ExtTh):
    '''
    Function where maturation occurs. Each timestep is computed.
    Input:
        params1 - species parameters. Type: dictionary
        c0 - initial chemical state. Type: 1d array
        table0 - initial species biomass and types. Type: Pandas table ['Parameter','Change','Number','Length']
        t - total maturation time. Type: int
        step - integration step to reach total maturation time. Type: float
        mutation_p - mutation probability. Type: float
    Output: 
        Updated species composition, with chemicals as an array and biomass as a table. 
    '''
    timestep = int(t/step)
    table_biom = table0
    cMed = c0

    for i in range(timestep):
            nCell, cMed, table_biom = cell_cycle_list(params1,cMed,table_biom,step,mutation_p,ExtTh)  
    
    return nCell, cMed, table_biom 


#def species_dynamics_table(params1,mutation_p,table0,t,c0):
#    '''
#    The dynamics of a multispecies community. Chemical mediated microbial interaction. 
#    Input:
#        params1 - dictionary with the parameters, K, alpha, beta, rho, r0 etc. Type: dictionary
#        mutation_p - the probability of mutation occuring. Type: int
#        table0 - the number of cells of each species. Type: pandas table [Parameter, Change, Number]
#        t - the time of interaction. Type: int
#        c0 - initial starting concentration of chemicals. Type: 1D array
#    Output:
#        The dynamics of chemically mediated interactions in a multi-species community for a time t.
#
#    '''
#    param_names = [i for i in params1 if i != 'num_spec'] # list of all names of the parameters in dictionary
#    ncols1 = get_ncols(params1,param_names) # getting the number of columns for each parameter, K, rho etc.
#    params_as_matrix = params_dict_to_matrix(params1,param_names) # combining all the columns into one big matrix
#    params_as_matrix.shape
#    n_c = params1['K'].shape[1]
#    n_s = params1['num_spec']

#    max_param = params_as_matrix.shape[1] # numer of columns, all parameters i.e dynamical quantities K, alpha, r0...

#    mutation0 = [mutation(table0[i],0,max_param,mutation_p) for i in range(len(table0))]
#    X = np.concatenate([sparse_dense(mutation0[i],max_param,params_as_matrix[i]) for i in range(len(mutation0))])

#    biomass0 = np.concatenate([mutation0[i]['Number'] for i in range(len(mutation0))])

#    dynamic_tot0 = params_matrix_to_dict(X, param_names, ncols1)

#    def dydt(t, sc): 
#        '''
#    Changing format to be solved by numerical integration. The format needs the ODE to have 2 variables
#    Input:
#        t - time span to solve the ODE
#        sc - initial condition values for all the species and chemicals (array of length n_s + n_c)
#    Output: 
#        The correct equation format to be solved in solve_ivp, where input is time and initial condition
#        '''
#        return sc_prime_std(t, sc, dynamic_tot0)

#    y0 = np.concatenate((biomass0, c0)) # initial conditions for dynamics calculations

    # sovle the ODE to get dynamics 
#    y = dynamics(dydt,y0,t)

#    nCell = y.y.T[-1:,:-n_c][0] # the number of each cell type after dynamics 
#    cMed = y.y.T[-1:,-n_c:][0] # chemical mediators after dynamics 

#    s_cut = cut_species(nCell,mutation0)
#    updated_table = [growth_number(s_cut[i],mutation0[i]) for i in range(len(mutation0))]
    
#    return cMed, nCell, updated_table


def maturation_table(params1,mutation_p,table0,tauf,dtau,c0):

    '''
    Dynamics of multi-species with specified timestep. Computes until final time dtau is reached.
    Input:
        params1 - dictionary with the parameters, K, alpha, beta, rho, r0 etc. Type: dictionary
        mutation_p - the probability of mutation occuring. Type: int
        table0 - the number of cells of each species. Type: pandas table [Parameter, Change, Number]
        tauf - the total time of chemically mediated species interaction. Type: int
        dtau - the timestep of each dynamics. Type: int
        c0 - initial starting concentration of chemicals. Type: 1D array
    '''
    step = int(tauf/dtau)
    cMed = c0
    table_biom = table0

    for i in range(step):
        cMed, nCell,table_biom = species_dynamics_table(params1,mutation_p,table_biom,dtau,cMed)
        #cMed, nCell,table_biom = species_dynamics_table_new(params1,mutation_p,table_biom,dtau,cMed)
    
    return cMed, nCell, table_biom


def NetworkConfig_Binomial(Nc,Nm,q):

    C = np.zeros([Nc,Nm])
    random_val = np.random.rand(Nc,Nm)
    C[random_val <= q] = 1

    return C

def DistInteractionStrengthMT_PA(Nc,Nm,ri0):
    '''
    % Interaction matrix based on strength probability distribution A
    % Nc: number of interacting species
    % ri0: maximum interaction strength
    % rpn: ratio of positive to negative interactions
        
    '''
    rint = ri0*(2*np.random.rand(Nc,Nm)-1)
    return rint



def DistInteractionStrengthMT_PB(Nc,Nm,ri0,fp):
    '''
    % Interaction matrix based on strength probability distribution B
    % Nc: number of interacting species
    % ri0: maximum interaction strength
    % fp: fraction for positive interactions
        
    '''
    rint = ri0 * np.random.rand(Nc,Nm) * np.sign(np.random.rand(Nc,Nm)-(1-fp))
    return rint


def interaction(y, Nc, r0, K, alpha, beta, rho_plus, rho_minus, kSat):

    breakpoint()

    S = np.reshape(y[:Nc], [Nc,1])
    C = np.reshape(y[Nc:], [len(y) - Nc, 1])

    denom = kSat + C.T
    rho_pos = rho_plus * np.reciprocal(denom)
    rho_min = np.reciprocal(kSat) * rho_minus
    rho_net = np.matmul(rho_pos + rho_min, C)
    S_ = (r0 + rho_net) * S

    # BELOW CODE WORKS, WE CAN DO CHUNKING LIKE THIS
    #S_chunk = S[0:3]
    #Ce = np.ones(len(S_chunk)) * C
    #A = np.reciprocal(Ce + kSat) * Ce
    #C_ = np.matmul(beta[:,0:len(S_chunk)] - (alpha[:,0:len(S_chunk)] * A), S_chunk)

    Ce =  np.ones(Nc) * C
    A = np.reciprocal(Ce + kSat) * Ce
    C_ = np.matmul(beta - (alpha * A), S)
    SC_ = np.vstack((S_, C_))

    return SC_

def dynamics_solver_(t, y, params):
    '''
    For solve_ivp friendly construction, make sure the parameters are input as a dictionary.
    Input:
        t - the time to compute dynamics. Type: int
        SC - the array of species biomass followed by chemical concentrations. Type: 2d array
        params - the parameters which include, r0, K, alpha, beta, rho_pos, rho_min etc. Type: dictionary
        Here, facilitative interactions, i.e 80 positive to 20 negative
        (to change this, simply change rho_plusB to rho_plus and rho_minusB to rho_minus which is the default 50/50)
    Output:
        The solve_ivp friendly form
    '''
    result = interaction(y,
                params["num_spec"],
                params["r0"],
                params["K"],
                params["alpha"],
                params["beta"],
                params["rho_plusB"],
                params["rho_minusB"],
                params['kSat'])
    return np.reshape(result, result.size).tolist()


def species_dynamics_array(t,params1,cMed,nCell,DilTh,ExtTh):
    '''
    Dynamics of chemical mediated interaction of multi species community. The biomass of species are array.
    Input:
        t - the time of simulation. Type: int
        params1 - dictionary with the parameters, K, alpha, beta, rho, 
        r0 etc. Type: dictionary
        cMed - initial starting concentration of chemicals. Type: 1d array
        nCell - the number of cells of each species. Type: 1d array
    Output:
        The dynamics of species interaction after time "t". The updated chemical concentration and species biomass.

    '''
    def dfdt(t, sc): 
        '''
    Changing format to be solved by numerical integration. The format needs the ODE to have 2 variables
    Input:
        t - time span to solve the ODE
        sc - initial condition values for all the species and chemicals (array of length n_s + n_c)
    Output: 
        The correct equation format to be solved in solve_ivp, where input is time and initial condition
        '''
        return dynamics_solver_(t, sc, params1)

    y0 = np.concatenate((nCell, cMed)) # initial conditions for dynamics calculations
 
    def event_(t,y0):
        return DilTh - sum(np.array(y0[:-len(cMed)]))
    
    event_.terminal = True

    y = solve_ivp(fun=dfdt, t_span=[0, t], method = 'LSODA', rtol = 10e-9, y0=y0, events=event_)  

    nCell = y.y.T[-1,:-len(cMed)] # the number of each cell type after dynamics 
    nCell[nCell<ExtTh] = 0
    cMed = y.y.T[-1,-len(cMed):] # chemical mediators after dynamics 

    return nCell, cMed


def dilution(Nr,Nc,c0,params1,TID,DilTh,ExtTh,tauf,name_):
    '''
    Simulation of dilution rounds, when total biomass reaches dilution threshold, the next round starts
    Input:

    '''
    cMed = c0
    cellRatio = 1 / Nc * np.ones(Nc)
    nCell = TID * cellRatio
    for i in range(Nr):
        cMed = TID / sum(nCell) * cMed # dilute the concentration of chemicals accourding to the new size of total species number, 
        nCell = TID * cellRatio
        nCell, cMed  =  species_dynamics_array(tauf,params1,cMed,nCell,DilTh,ExtTh)
        # dilution of chemicals and species numbers 
        cellRatio = 1 / sum(nCell) * nCell # new cell ratio, where we dilute after the total number of cells is bigger than dilution threshold
        result_save_(f'{name_}_{i}',nCell)
    return nCell, cMed

### The same as before but instead of stopping at an event, this is time based simulation, where dynamics occurs until specified time ###

def species_dynamics_array_time(t,params1,cMed,nCell,DilTh,ExtTh):
    '''
    Dynamics of chemical mediated interaction of multi species community. The biomass of species are array.
    Here this version does not have the event functionality in the solve_ivp. The dynamics will be computed accourding to final set time
    Input:
        t - the time of simulation. Type: int
        params1 - dictionary with the parameters, K, alpha, beta, rho, 
        r0 etc. Type: dictionary
        cMed - initial starting concentration of chemicals. Type: 1d array
        nCell - the number of cells of each species. Type: 1d array
    Output:
        The dynamics of species interaction after time "t". The updated chemical concentration and species biomass.

    '''
    def dfdt(t, sc): 
        '''
    Changing format to be solved by numerical integration. The format needs the ODE to have 2 variables
    Input:
        t - time span to solve the ODE
        sc - initial condition values for all the species and chemicals (array of length n_s + n_c)
    Output: 
        The correct equation format to be solved in solve_ivp, where input is time and initial condition
        '''
        return dynamics_solver_(t, sc, params1)

    y0 = np.concatenate((nCell, cMed)) # initial conditions for dynamics calculations
 
    def event_(t,y0):
        return DilTh - sum(np.array(y0[:-len(cMed)]))
    
    event_.terminal = False

    y = solve_ivp(fun=dfdt, t_span=[0, t],method='LSODA', rtol = 10e-9,y0=y0)  

    nCell = y.y.T[-1,:-len(cMed)] # the number of each cell type after dynamics 
    nCell[nCell<ExtTh] = 0
    cMed = y.y.T[-1,-len(cMed):] # chemical mediators after dynamics 
    return nCell, cMed, y.t[-1]


def dynamics_cycle_time(params1,DilTh,ExtTh,tauf,dtau,c0,nCell0):
   '''
    Dynamics of multi-species for multiple timestep. Where instead of interaction persisting until biomass reaching threshold,
    Here there is final 'tauf' for which the simulation runs until, with the step size of dtau
    (Can be used to choose a partiuclar timestep or when not using "events" in solve_ivp.)
    Input:
        params1 - dictionary with the parameters, K, alpha, beta, rho, r0 etc. Type: dictionary
        DilTh - coculture dilution threshold. When total cells reach this number, dilution occurs. Type: int 
        tauf - the total time of chemically mediated species interaction. Type: int
        dtau - the timestep of each dynamics. Type: int
        cMed - initial starting concentration of chemicals. Type: 1d array
        nCell - the number of cells of each species. Type: 1d array
    Output:
        The dynamics until reaching final time dtau or stopping of otal biomass reaches dilution threshold. 
    '''
   step = int(tauf/dtau)
   cMed = c0
   nCell = nCell0
   for i in range(step):
        nCell,cMed,time = species_dynamics_array_time(dtau,params1,cMed,nCell,DilTh,ExtTh)
   return nCell, cMed, time


def dilution_time(Nr,Nc,c0,params1,TID,DilTh,ExtTh,tauf,dtau,name_):
    '''
    Simulation of dilution rounds, when total biomass reaches dilution threshold, the next round starts
    Input:

    '''
    cMed = c0
    cellRatio = 1 / Nc * np.ones(Nc)
    nCell = TID * cellRatio
    for i in range(Nr):
        cMed = TID / sum(nCell) * cMed # dilute the concentration of chemicals accourding to the new size of total species number, 
        nCell = TID * cellRatio
        nCell, cMed, time = dynamics_cycle_time(params1,DilTh,ExtTh,tauf,dtau,cMed,nCell)
        # dilution of chemicals and species numbers 
        cellRatio = 1 / sum(nCell) * nCell # new cell ratio, where we dilute after the total number of cells is bigger than dilution threshold
        result_save_(f'{name_}_{i}',nCell)
    return nCell, cMed, time

### End of the time based simulation ###


### Time based simulation for Tree structure ###


def params_tree(params,ExtTh):

    params1 = {}
    params1['num_spec'] = params['num_spec']
    params1['Nm'] = params['Nm']
    params1['r0'] = params['r0'] 
    params1['K'] = params['K']
    params1['alpha'] = params['alpha'].T
    params1['beta'] = params['beta'].T
    params1['rho_plus'] = params['rho_plus']
    params1['rho_minus'] = params['rho_minus']
    params1['kSat'] = params['kSat']
    params1['ExtTh'] = ExtTh

    return params1


def sparse_array(spec_sparse,array_cols,min_param,max_param,ancestor_param):
    
    '''
    Function which converts table into the sparse csr_matrix form
    Input:
        table - table to be stored as sparse csr_matrix form. Type: pandas table [Parameter,Change,Number,Length]
        min_param - the first parameter index. Type: int (typically 0).
        max_param - the maximum parameter value, the total number of parameters. Type: int.
        initial - initial conditions with initial values for all parameters, max_param number. Type: 1d array.
    Output:
        A csr_matrix form which stores the content of the table  
    '''
    
    #breakpoint()

    col = np.append(np.arange(0,max_param),spec_sparse[:,array_cols['Parameter']]).astype('int64')
    row = np.append(np.zeros(max_param),np.arange(spec_sparse.shape[0])).astype('int64')
    data = np.append(ancestor_param,spec_sparse[:,array_cols['Change']])
    
    #col2 = np.concatenate([np.arange(0,max_param),table['Parameter']]) # columns are the parameter values
    #row2 = np.concatenate([np.int64(np.zeros(max_param)), np.arange(len(table))])
   # data2 = np.concatenate([initial_condition, table['Change']]) # the changes from the table usually after mutation
    
    return csr_matrix((data,(row,col)))

def sparse_dense_array(spec_sparse,array_cols,max_param,ancestor_param):
    '''
    Combining the sparse to dense operation as one step to simplify later parts.
    Input: 
        table - table with updated population sizes and rows. Type: pandas table [Parameter,Change,Number,Length].
        max_param - the total number of parameters, based on number of chemicals. Type: int
                    where the columns of the matrix represent all the parameters: K, r0, alpha, beta, rho_plus, rho_minus
        ancestor_row - initial value of each parameter for the ancestral population. Type: 1d array
                       Single row with the total number of parameters column.
    Output: 
        The dense matrix with updated population sizes after mutation.
    '''
    # create sparse matrix from the table with (ancestor + 1 round of mutation)

    Z = sparse_array(spec_sparse,array_cols,0,max_param,ancestor_param) # creating a sparse form from the resulting mutation 
    X = dense(Z) # total dense matrix for species 1 after 1 round of mutation

    return X

# vectorised calculation of 
def mutation_distribution_n(no_mutants):
    '''
    Vectorised calculation of changes to parameter values, nased on a normal distribution.

    Parameters
    ----------
    no_mutants : TYPE float64 (might be single-element numpy array, not sure)
        DESCRIPTION. Number of mutants, generated through one mutation step, from all 
        subpopulations of all species (sum of num_mutation) 

    Returns
    -------
    TYPE np.array float64
        DESCRIPTION. array of changes to parameter values for each mutant (size = no_mutants), 
        sampled from a normal distribution (mu = 0, sigma = 1), rounded to 2.d.p

    '''
    return np.round(np.random.normal(size=no_mutants),2)


def create_mutants_np(min_param,max_param,mutation_distribution_n,num_mutation):
    '''
    Create dataframe of new mutants generated during the mutation step. Rows take the form
        ["Parameter","Change","Number"].
    This is added to the main dataframe.
    (NOTE - This function used to contain a step that removed any mutation_change = 0. This is quite unlikely (e.g. P(0) = 0.004 in 100000),
     so I have removed it, as it slows down model performance by adding conditions, and we have to update num_mutation and ancest_row).
    I think cleaning of these rows can be done at the end of simulations.

    Parameters
    ----------
    min_param : TYPE int32
        DESCRIPTION. Minimum parameter index. Typically 0.
    max_param : TYPE int32
        DESCRIPTION. Index of final parameter.
    mutation_distribution_n : TYPE function
        DESCRIPTION. Calls mutation_distribution_n, which calculates changes to parameters.
    num_mutation : TYPE np.array of int
        DESCRIPTION. array of number of mutations per subpopulation.

    Returns
    -------
    mutant_df : TYPE pd.DataFrame
        DESCRIPTION. Dataframe tracking the change in parameter values for the mutations. Columns =
        ["Parameter","Change","Number"]. Also includes closeout row. Double indexed, first index = 
        index of the ancestral subpopulation, second index = standard indexing. Allows mutation rows 
        be inserted in the correct place with one implementation of pd.concat.

    '''
    
    no_mutants = np.sum(num_mutation) # total no. mutants from no. mutants per ancestral subpopulation
    
    select_parameter = np.random.randint(min_param,max_param,no_mutants) # vectorised selection of parameters to mutate in the new mutants
    mutation_change = mutation_distribution_n(no_mutants) # vectorised calculation of change in selected parameters in mutants
    
    mutant_dat = np.repeat(np.vstack((select_parameter,mutation_change)),2,axis=1) # stack corresponding parameter and change 
    #   per mutant together, but duplicate each row so we can calculate the closeout rows.
    mutant_dat[1,1::2] = -mutant_dat[1,1::2] # closeout row ['Change'] = - mutant row ['Change']
    #mutant_dat = np.vstack((mutant_dat,np.tile(np.array([1,0]),int(mutant_dat.shape[1]/2)))).T
    mutant_dat = np.vstack((mutant_dat,np.tile(np.array([1,0]),len(num_mutation)))).T # add new column for ['Number']. New mutant rows
    #   are filled with 1s, because one new mutant is generated from the ancestral subpop. Closeout rows are filled with 0s.
        
    
    indexer = [np.repeat(np.arange(len(num_mutation)),num_mutation*2),
               np.arange(1,mutant_dat.shape[0]+1)] # indexing that allows mutant and closeout rows to be inserted in the correct
    #   place. First index = index on ancestral subpop. 2nd index = standard indexing so after the new rows are inserted in the
    #   correct place during pd.concat(), rows remain in the correct order.
    mutant_df = pd.DataFrame(mutant_dat,columns=["Parameter","Change","Number"],
                             index=indexer) # convert np.array to dataframe
    
    return mutant_df


def mutation_df(big_table,min_param,max_param,mutation_p,ancest_row):
    '''
    Carry out mutation on existing subpopulations of species.

    Parameters
    ----------
    big_table : TYPE pd.DataFrame
        DESCRIPTION. Dataframe of ancestral subpopulations, taking the form ['Parameter','Change','Number'].
        'Parameter' = parameter changed from ancestor, 'Change' = change in value to said 'Parameter',
        'Number' = biomass/population size of subpopulation.
    min_param : TYPE int32
        DESCRIPTION. Minimum parameter index. Typically 0.
    max_param : TYPE int32
        DESCRIPTION. Index of final parameter.    
    mutation_p : TYPE float64
        DESCRIPTION. Probability of mutation occuring. No. mutations per subpop = mutation_p * subpop_Number.
    ancest_row : TYPE np.array int32
        DESCRIPTION. Keeps track of the indices of the original species parameter values. Used in sparse-to-dense matrix
            conversion.

    Returns
    -------
    new_table : TYPE pd.DataFrame
        DESCRIPTION. Updated big_table with new mutants.

    '''
    
    #breakpoint()
    
    num_mutation = mutation_freq(mutation_p,big_table['Number'].astype('int32'),big_table.shape[0]) # calculate no. new mutants
    #   per ancestral subpopulation.
    
    big_table['Number'] -= num_mutation # remove new mutants from ancestral subpopulation
    ancest_row[1:] += np.cumsum(np.multiply(num_mutation[:-1],2)) # update ancest_row with new position of species ancestors in the new table.
    #   (shouldn't need to return , but check)
    
    #mutants_to_add, final_mutation = create_mutants_np(min_param,max_param,mutation_distribution,num_mutation)
    mutants_to_add = create_mutants_np(min_param,max_param,mutation_distribution_n,num_mutation) # generate dataframe of new mutants
    
    new_table = pd.concat([big_table,mutants_to_add]).sort_index() # add new_mutants to big_table, inserted into their correct place using indexing
    new_table.index = [np.arange(new_table.shape[0]),np.arange(new_table.shape[0])] # update new_table index to standard multindexing.
    
    return new_table
    
                    
def mutation_array(table_to_array,array_cols,min_param,max_param,mutation_p):
    
    '''
  A function which adds mutation to an ancestor 
  Input: 
      table - the starting table with ancestor, with columns: parameter, change and number of cells  
      min_param - the 1s parameter index
      max_param - the last parameter index, total number of parameters
      mutation_p - mutation probability
  Return:
      A table with added mutations to random parameters. Branching from ancestor to different subpopulation
      
    '''
    #breakpoint()
    
    #global table_to_array
    
    for arry_ind in range(len(table_to_array)): 
    
        spec_sparse = table_to_array[arry_ind]
    
        num_mutation = mutation_freq(mutation_p,spec_sparse[:,array_cols['Number']].astype('int32'),spec_sparse.shape[0])
        #num_mutation = mutation_freq(0.01,array[:,array_cols['Number']].astype('int32'),array.shape[0])
     
        og_array_rows = spec_sparse.shape[0]
        
        for i in reversed(range(og_array_rows)):
            
            if num_mutation[i] != 0:
                
                for each in range(num_mutation[i]):
            
                    parameter, change = rand_mut_param(min_param,max_param,mutation_distribution,1)
                    
                    if change != 0:
                        
                        spec_sparse[i,array_cols['Number']] -= 1
                        
                        # add new subpopulations
                        new_subpop = np.array([parameter,change,1,spec_sparse[i,array_cols['Length']]])
                        closeout_row = np.array([parameter,-change,0,0])
                        
                        spec_sparse = np.vstack((spec_sparse[:i+1,:],new_subpop,closeout_row,
                                                spec_sparse[i+1:,:]))
                        
        table_to_array[arry_ind] = spec_sparse
    
    #return array;
    


def dynamics_solver_tree(t,y,params):
    '''
    For solve_ivp friendly construction, make sure the parameters are input as a dictionary.
    Input:
        t - the time to compute dynamics. Type: int
        SC - the array of species biomass followed by chemical concentrations. Type: 2d array
        params - the parameters which include, r0, K, alpha, beta, rho_pos, rho_min etc. Type: dictionary
    Output:
        The solve_ivp friendly form
    '''
    result = interaction(y,
                params["num_spec"],
                params["r0"],
                params["K"],
                params["alpha"].T,
                params["beta"].T,
                params["rho_plus"],
                params["rho_minus"],
                params['kSat'])
    return np.reshape(result, result.size).tolist()

#@njit
def dynamics_calc_chunk(t,s_chunk,dense_chunk,chem,parm_ind,kSat):
    '''
    Calculate d(Population)/dt and d(Chemicals)/dt

    Parameters
    ----------
    t : TYPE int or float
        DESCRIPTION. total time for ODE solver
    s_chunk : TYPE np.array of float64
        DESCRIPTION. Subpopulation biomass/population sizes in chunk.
    chem :  np.array of float64
        DESCRIPTION. chemical mediator concentrations
    dense_chunk : TYPE np.array of float64
        DESCRIPTION. dense matrix of chunk of subpopulations. Has parameter values and population size.
    parm_ind : TYPE np. array of ind32
        DESCRIPTION. columns of dense_chunk corresponding to different parameters.

    Returns
    -------
    result : TYPE 2 x np.array of float64.
        DESCRIPTION. d(Population)/dt and d(Chemicals)/dt

    '''
    
    
    # What will be the tidy version
    #r0,K,alpha,beta,rho_plus,rho_minus = np.array_split(dense_chunk,parm_ind)[:-1]
    # or something tidier where we extract names and values (might be hard with numba)
    
    
    r0 = dense_chunk[:,0]
    K = dense_chunk[:,1:8]
    alpha = dense_chunk[:,8:15]
    beta = dense_chunk[:,15:22]
    rho_plus = dense_chunk[:,22:29]
    rho_minus = dense_chunk[:,29:]
    
    result = interaction_chunk(s_chunk,chem,r0,K,alpha,beta,rho_plus,rho_minus,kSat)
                
                #params["r0"],
                #params["K"],
                #params["alpha"].T,
                #params["beta"].T,
                #params["rho_plus"],
                #params["rho_minus"],
                #params['kSat'])
    #return np.reshape(result, result.size).tolist()
    
    return result

#@njit
def interaction_chunk(s_chunk, chem, r0, K, alpha, beta, rho_plus, rho_minus, kSat):
    '''
    

    Parameters
    ----------
    s_chunk : TYPE np.array of float64
        DESCRIPTION. Subpopulation biomass/population sizes (not calculated from dense matrix every time_step, I think this would worsen runtime).
    chem :  np.array of float64
        DESCRIPTION. chemical mediator concentrations
    ... : parameters

    Returns
    -------
    dS : TYPE np.array of float64.
        DESCRIPTION. d(Population)/dt
    dC :  np.array of float64.
        DESCRIPTION. d(Chemicals)/dt

    '''
    
    denom = kSat + chem.T
    rho_pos = rho_plus * np.reciprocal(denom)
    rho_min = np.reciprocal(kSat) * rho_minus
    #rho_net = np.matmul(rho_pos + rho_min, chem)
    rho_net = (rho_pos + rho_min) @ chem
    dS = (r0 + rho_net) * s_chunk
    
    # this might not be correct anymore
    Ce =  np.ones(len(s_chunk)) * chem
    A = np.reciprocal(Ce + kSat) * Ce
    #dC = np.matmul(beta - (alpha * A), s_chunk)
    dC = (beta - (alpha * A)) @ s_chunk
    
    return dS, dC

###############
# this is one whole function
def csum(d_m):
    '''
    Cumsum per species for sparse-dense array conversion. Function is vectorised.

    Parameters
    ----------
    d_m : TYPE np.array
        DESCRIPTION. # sparse matrix form (but not actually sparse matrix type) of changes to parameter values 
            (filled with 0s).

    Returns
    -------
    TYPE np.array
        DESCRIPTION. cumulative sum of d_m, creates dense matrix

    '''
    
    return np.cumsum(d_m)

v_func = np.vectorize(csum) # above sum is vectorosed. This is the function implemented in sparse-to-dense array conversion.
####################

#@njit
def sdf_to_dense(df,min_param,max_param,ancest_row,ancest_parms):
    '''
    Converts pd.Dataframe of the manually-coded sparse dataframe to the dense np.array of parameter values.
        Aka sparse-to-dense conversion.

    Parameters
    ----------
    df : TYPE pd.DataFrame
        DESCRIPTION. Manually-coded sparse array to be converted.
    min_param : TYPE int32
        DESCRIPTION. Minimum parameter index. Typically 0.
    max_param : TYPE int32
        DESCRIPTION. Index of final parameter.    
    ancest_row : TYPE np.array int32
        DESCRIPTION. Keeps track of the indices of the original species parameter values. Used in sparse-to-dense matrix
            conversion.
    ancest_parms : TYPE np.array of float64
        DESCRIPTION. Parameter values of ancestral species. Used for cumsum to calculate parameters of all subpopulations.

    Returns
    -------
    TYPE np.array
        DESCRIPTION. Dense matrix of parameter values. Used in dynamics calculations.

    '''
    
    breakpoint()
    
    dense_mat = np.zeros((df.shape[0],(max_param-min_param)))
    
    dense_mat[np.arange(df.shape[0]),(df[:,0].astype('int32')-1)] = df[:,1]
    dense_mat[ancest_row,:] = ancest_parms
    
    #dense_mat[np.arange(df_to_np.shape[0]),(df_to_np[:,0]-1).astype('int32')] = df_to_np[:,1]
    #dense_mat[ancest_r,:] = ancest_parms
    
    # cumsum
    dense_mat = np.vsplit(dense_mat,ancest_row[1:])

    return np.concatenate(v_func(dense_mat),axis=0)

# this function is incorrect, ancest_row term is not right
def sdf_to_dense_alt(df,min_param,max_param,ancest_row,ancest_parms):
    '''
    Converts pd.Dataframe of the manually-coded sparse dataframe to the dense np.array of parameter values.
        Aka sparse-to-dense conversion.

    Parameters
    ----------
    df : TYPE pd.DataFrame
        DESCRIPTION. Manually-coded sparse array to be converted.
    min_param : TYPE int32
        DESCRIPTION. Minimum parameter index. Typically 0.
    max_param : TYPE int32
        DESCRIPTION. Index of final parameter.    
    ancest_row : TYPE np.array int32
        DESCRIPTION. Keeps track of the indices of the original species parameter values. Used in sparse-to-dense matrix
            conversion.
    ancest_parms : TYPE np.array of float64
        DESCRIPTION. Parameter values of ancestral species. Used for cumsum to calculate parameters of all subpopulations.

    Returns
    -------
    TYPE np.array
        DESCRIPTION. Dense matrix of parameter values. Used in dynamics calculations.

    '''
    
    breakpoint()
    
    dense_mat = np.zeros((df.shape[0],(max_param-min_param)))
    
    dense_mat[np.arange(df.shape[0]),(df['Parameter']-1)] = df['Change']
    dense_mat[ancest_row,:] = ancest_parms

    return np.cumsum(dense_mat,axis=0)

#@njit
def convert_and_dynamics(chunk,t,s_chunk,chem,min_param,max_param,ancest_row,ancest_parms,parm_ind,kSat):
    
    dense_matrix = sdf_to_dense(chunk,min_param,max_param,ancest_row,ancest_parms)

    dS_part, dC_part = dynamics_calc_chunk(t,s_chunk,dense_matrix,chem,parm_ind,kSat)
    
    return dS_part, dC_part

#@njit
def apply_cnvrt_dnmcs(chunks,t,spec,chem,min_param,max_param,
                                        a_r_chunked,params_as_matrix,parm_ind,kSat):
    
    dS_all = []
    dC_all = np.zeros(chem.shape)
    
    rows_prev_chunk = 0 
    
    for i in range(len(chunks)):
        
        df_chunk = chunks[i]
        ancest_r_chunk = a_r_chunked[i]
        
        ancest_r_chunk -= rows_prev_chunk
        
        s_chunk = spec[rows_prev_chunk:(rows_prev_chunk+df_chunk.shape[0])]
        
        dS_chunk, dC_chunk = convert_and_dynamics(df_chunk,t,s_chunk,chem,min_param,max_param,
                                                ancest_r_chunk,params_as_matrix,parm_ind,kSat)
        dS_all.append(dS_chunk)
        dC_all += dC_chunk
        
        rows_prev_chunk += df_chunk.shape[0]
    
    return np.concatenate(np.asarray(dS_all),dC_all)


#def species_dynamics_evo(params1,mutation_p,table0,t,c0):
    '''
def species_dynamics_evo(dtau,
                         biomass_table,ancest_row,
                         parms_to_evolve,e_parm_ind,fixed_parm,f_parm_ind,mutation_p,
                         cMed):
    #'''
    

    #Parameters
   # ----------
    #dtau : TYPE
    #    DESCRIPTION.
    #biomass_table : TYPE
    #    DESCRIPTION.
    #ancest_row : TYPE
    #    DESCRIPTION.
    #parms_to_evolve : TYPE
    #    DESCRIPTION.
    #e_parm_ind : TYPE
    #    DESCRIPTION.
    #fixed_parm : TYPE
    #    DESCRIPTION.
    #f_parm_ind : TYPE
    #    DESCRIPTION.
    #mutation_p : TYPE
    #    DESCRIPTION.
    #cMed : TYPE
    #    DESCRIPTION.

    #Returns
    #-------
    #dSCdt : TYPE
    #    DESCRIPTION.

    '''
    
    breakpoint()
    
    # Convert parameters into matrix form - keep for now to use in calculations, will eventually be a function argument
    
    #param_names = [i for i in params1 if i !='num_spec' and i !='Nm' and i != 'kSat' and i!='ExtTh'] # list of all names of the parameters in dictionary
    #ncols1 = get_ncols(params1,param_names) # getting the number of columns for each parameter, K, rho etc.
    #params_as_matrix = params_dict_to_matrix(params1,param_names) # combining all the columns into one big matrix

    #n_c = params1['K'].shape[1]
    #ExtTh = params1['ExtTh']

    #min_param = 0
    #max_param = params_as_matrix.shape[1] # numer of columns, all parameters i.e dynamical quantities K, alpha, r0...
    
    #parm_ind = ...
    
    ################################

    # Convert table0 into big_table for my new function, big_table will eventually be a function argument
       
    #big_table = pd.concat([df for df in table0]).reset_index(drop = True)
    #big_table = big_table.drop(columns='Length')
    
    #ancest_row = np.array(big_table.index) # record index of all species ancestors, will also be a function argument
    # remember to return this for species_dynamics_table_new
    
    
    ################################
    
    min_param = 0
    max_param = parms_to_evolve.shape[1]
    

    ################################
    
    # Implement evolution
    
    biomass_table = mutation_df(biomass_table,min_param,max_param,0.01,ancest_row) # big_table and ancest_row are updated
    # ancest_row should be updated within
    
    #big_table, ancest_row = mutation_df(big_table,0,max_param,0.01,ancest_row) # big_table and ancest_row are updated
    
    ##########################################
    
    # Dynamics calculations
     
    # Chunked dynamics calculations
    #   Only implement chunking if absolutely necessary
    
    chunk_size = 5000 # for example, will be a function argument
    
    # algorithm with chunking
    if (ancest_row >= chunk_size).any():
        
        chunk_rng = np.arange(0,ancest_row[-1],chunk_size)
        chunks = np.array_split(biomass_table,ancest_row[np.searchsorted(ancest_row,chunk_rng)])
        
        a_r_chunked = np.array_split(ancest_row,ancest_row[np.searchsorted(ancest_row,chunk_rng)])
        # this is maybe correct
        
        # I can't have this function being re-compiled every call of species_dynamics_table_new, so define elsewhere
        def dDynamics_dt_chunk(t,y,min_param,max_param,ancest_row,params_as_matrix,parm_ind,kSat):
            
            spec = y[:(biomass_table.shape[0]+1)]
            chem = y[(biomass_table.shape[0]+1):]
            
            #dS_all, dC_all = zip(*[convert_and_dynamics(df_chunk,t,spec,chem,min_param,max_param,
           #                                             ancest_row,params_as_matrix,parm_ind,kSat) for df_chunk in chunks])

            #dS_all = np.concatenate(dS_all)
            #dC_all = np.sum(dC_all)
            
            #dSCdt = np.concatenate((dS_all,dC_all))
            
            dSCdt = apply_cnvrt_dnmcs(chunks, t, spec, chem, min_param, max_param, ancest_row, params_as_matrix, parm_ind, kSat)
            
            return dSCdt
            
        dynamics_sol = solve_ivp(fun=dDynamics_dt_chunk, t_span=[0,t], method = 'LSODA', rtol = 10e-9, y0 = np.concatenate((big_table['Number'],c0)))
      
    # algorithm without chunking    
    else: 
        
        dense_matrix = sdf_to_dense_alt(biomass_table, min_param, max_param, ancest_row, params_as_matrix)

        def dDynamics_dt(t,y,dense_matrix,parm_ind,kSat):
            
            spec = y[:(biomass_table.shape[0]+1)]
            chem = y[(biomass_table.shape[0]+1):]
         
            dSCdt = np.concatenate(dynamics_calc_chunk(t,spec,chem,dense_matrix,parm_ind,kSat))
            
            return dSCdt
            
        dynamics_sol = solve_ivp(fun=dDynamics_dt, t_span=[0,t], method = 'LSODA', rtol = 10e-9, y0 = np.concatenate((species0,c0)))
            
          
    
    
    
    
    
    
    

        
    nCell = y.y.T[-1:,:-n_c][0] # the number of each cell type after dynamics 
    nCell[nCell<ExtTh] = 0
    cMed = y.y.T[-1:,-n_c:][0] # chemical mediators after dynamics 

    #s_cut = cut_species(nCell,mutation0)
    #updated_table = [growth_number(s_cut[i],mutation0[i]) for i in range(len(mutation0))]
    
    
    #sparse-dense matrix conversion on numpy array
    #X2 = np.concatenate([sparse_dense_array(table_to_array[i],array_cols,max_param,params_as_matrix[i]) for 
    #                     i in range(len(table_to_array))])
 
#    X = np.concatenate([sparse_dense(mutation0[i],max_param,params_as_matrix[i]) for i in range(len(mutation0))])
   
#    dynamics_format = params_matrix_to_dict(X, param_names, ncols1)
#    dynamics_format['kSat'] = params1['kSat']
#    
 #   def dydt(t, sc): 
#        '''
#    Changing format to be solved by numerical integration. The format needs the ODE to have 2 variables
#    Input:
#        t - time span to solve the ODE
#        sc - initial condition values for all the species and chemicals (array of length n_s + n_c)
#    Output: 
#        The correct equation format to be solved in solve_ivp, where input is time and initial condition
#        '''
 #       return dynamics_solver_tree(t, sc, dynamics_format)

 #   y0 = np.concatenate((biomass0, c0)) # initial conditions for dynamics calculations
  
    #def event_(t,y0):
    #    return DilTh - sum(np.array(y0[:-n_c]))
    
    #event_.terminal = False

    #y = solve_ivp(fun=dydt, t_span=[0, t], method = 'LSODA', rtol = 10e-9, y0=y0, events=event_)
#    y = solve_ivp(fun=dydt, t_span=[0, t], method = 'LSODA', rtol = 10e-9, y0=y0)

#    nCell = y.y.T[-1:,:-n_c][0] # the number of each cell type after dynamics 
#    nCell[nCell<ExtTh] = 0
#    cMed = y.y.T[-1:,-n_c:][0] # chemical mediators after dynamics 

#    s_cut = cut_species(nCell,mutation0)
 #   updated_table = [growth_number(s_cut[i],mutation0[i]) for i in range(len(mutation0))]
    
    #breakpoint()

    #return cMed, nCell, updated_table


#def species_dynamics_standard(params1,mutation_p,table0,t,c0):

def species_dynamics_table(params1,mutation_p,table0,t,c0):
    '''
    The dynamics of a multispecies community. Chemical mediated microbial interaction. 
    Input:
        params1 - dictionary with the parameters, K, alpha, beta, rho, r0 etc. Type: dictionary
        mutation_p - the probability of mutation occuring. Type: int
        table0 - the number of cells of each species. Type: pandas table [Parameter, Change, Number]
        t - the time of interaction. Type: int
        c0 - initial starting concentration of chemicals. Type: 1D array
    Output:
        The dynamics of chemically mediated interactions in a multi-species community for a time t.

    '''
    
    
    param_names = [i for i in params1 if i !='num_spec' and i !='Nm' and i != 'kSat' and i!='ExtTh'] # list of all names of the parameters in dictionary
    ncols1 = get_ncols(params1,param_names) # getting the number of columns for each parameter, K, rho etc.
    params_as_matrix = params_dict_to_matrix(params1,param_names) # combining all the columns into one big matrix

    n_c = params1['K'].shape[1]
    ExtTh = params1['ExtTh']

    max_param = params_as_matrix.shape[1] # numer of columns, all parameters i.e dynamical quantities K, alpha, r0...

    
    mutation0 = [mutation(table0[i],0,max_param,mutation_p) for i in range(len(table0))]
    
    X = np.concatenate([sparse_dense(mutation0[i],max_param,params_as_matrix[i]) for i in range(len(mutation0))])
    biomass0 = np.concatenate([mutation0[i]['Number'] for i in range(len(mutation0))])

    dynamics_format = params_matrix_to_dict(X, param_names, ncols1)
    dynamics_format['kSat'] = params1['kSat']

    def dydt(t, sc): 
        '''
    Changing format to be solved by numerical integration. The format needs the ODE to have 2 variables
    Input:
        t - time span to solve the ODE
        sc - initial condition values for all the species and chemicals (array of length n_s + n_c)
    Output: 
        The correct equation format to be solved in solve_ivp, where input is time and initial condition
        '''
        return dynamics_solver_tree(t, sc, dynamics_format)

    y0 = np.concatenate((biomass0, c0)) # initial conditions for dynamics calculations
  
    #def event_(t,y0):
    #    return DilTh - sum(np.array(y0[:-n_c]))
    
    #event_.terminal = False

    #y = solve_ivp(fun=dydt, t_span=[0, t], method = 'LSODA', rtol = 10e-9, y0=y0, events=event_)
    y = solve_ivp(fun=dydt, t_span=[0, t], method = 'LSODA', rtol = 10e-9, y0=y0)

    nCell = y.y.T[-1:,:-n_c][0] # the number of each cell type after dynamics 
    nCell[nCell<ExtTh] = 0
    cMed = y.y.T[-1:,-n_c:][0] # chemical mediators after dynamics 

    s_cut = cut_species(nCell,mutation0)
    updated_table = [growth_number(s_cut[i],mutation0[i]) for i in range(len(mutation0))]
    
#    breakpoint()

    return cMed, nCell, updated_table


def maturation_table(params1,mutation_p,table0,c0,tauf,dtau):

    '''
    Dynamics of multi-species with specified timestep. Computes until final time dtau is reached.
    Input:
        params1 - dictionary with the parameters, K, alpha, beta, rho, r0 etc. Type: dictionary
        mutation_p - the probability of mutation occuring. Type: int
        table0 - the number of cells of each species. Type: pandas table [Parameter, Change, Number]
        tauf - the total time of chemically mediated species interaction. Type: int
        dtau - the timestep of each dynamics. Type: int
        c0 - initial starting concentration of chemicals. Type: 1D array
    '''
    step = int(tauf/dtau)
    biomass_table = table0
    cMed = c0
    for i in range(step):
        cMed, nCell,biomass_table = species_dynamics_table(params1,mutation_p,biomass_table,dtau,cMed)
    
    return cMed, nCell, biomass_table



def dilution_table(Nr,Nc,c0,table0,params1,TID,mutation_p,tauf,dtau,name_):
    '''
    Simulation of dilution rounds, when total biomass reaches dilution threshold, the next round starts
    Input:

    '''
    cMed = c0
    cellRatio = 1 / Nc * np.ones(Nc)
    nCell = TID * cellRatio
    biomass_table = table0
    for i in range(Nr):
        cMed = TID / sum(nCell) * cMed # dilute the concentration of chemicals accourding to the new size of total species number, 
        nCell = TID * cellRatio
        cMed, nCell, biomass_table = maturation_table(params1,mutation_p,biomass_table,cMed,tauf,dtau)
        # dilution of chemicals and species numbers 
        cellRatio = 1 / sum(nCell) * nCell # new cell ratio, where we dilute after the total number of cells is bigger than dilution threshold
        result_save_(f'{name_}_{i}',nCell)
    return nCell, cMed, biomass_table

### End of time based simulation for tree structure 3 ###

def contribution(Ne,cellRatio):
    '''
    The Cmp as a percentage of each cell type contributes to the total community
    Input:
        Ne - picking the indexes of the non-zero cell from each species. Type - 1d array
        cellRatio -  new cell ratio, where we dilute after the total number of cells is bigger than dilution threshold. Type: 1d array
    Output:
        The non-zero contribution of each cell types to the total communuty 
    '''
    
    Cmp = cellRatio[Ne] # select the cell ratios where this species member satisfies the condition 
    if Cmp.sum() > 0:
        Cmp_sum = np.zeros(len(Cmp))
        Cmp_sum[:] = Cmp.sum()
        Cmp = Cmp/Cmp_sum
        return Cmp


def coexistence(Nr,Nc,params1,TID,DilTh,ExtTh,tauf):

    cMed = np.zeros(params1["Nm"])
    cellRatio = 1 / Nc * np.ones(Nc) 
    nCell0 = cellRatio *TID
    nCell = nCell0
    for i in range(Nr):
        cMed = TID / sum(nCell) * cMed # dilute the concentration of chemicals accourding to the new size of total species number, 
        nCell = TID * cellRatio
        nCell, cMed  =  species_dynamics_array(tauf,params1,cMed,nCell,DilTh,ExtTh)
        # dilution of chemicals and species numbers 
        cellRatio = 1 / sum(nCell) * nCell # new cell ratio, where we dilute after the total number of cells is bigger than dilution threshold

    id = np.linspace(0,Nc-1,Nc)
    Ne0 = id[nCell>0.5].astype('int')
    cmp0 = contribution(Ne0,cellRatio)

    nGen = np.log(DilTh/TID)/np.log(2)
    r = (nCell/nCell0)**(20/nGen) 
    stp = (r > abs(0.9*max(r)))

    Ne = id[stp].astype('int')
    cmp = contribution(Ne,cellRatio)

    return cmp0, cmp

def open_directory(name):
    '''
    Function to access direcotry to access files. To get directory name option+command+Copy
    Input:
        name - the name of the directory. To get this on MacOS press option+command+Copy. Type: string
    Output:
        The access to the files inside the given directory
    '''
    path = os.path.expanduser(f'{name}')
    return path


def result_save_(name, results):
    '''
    A function which saves the results from simulation. For example Adult data, which contains [table, biomass, Product]
    Input:
        name - the designated name of the file. The input should be in terms of string, i.e 'name'
        i - the cycle index. In the form of integer 
        results - the simulation results i.e the variable which contains all Adult from a single cycle
    Output:
        A folder with the name "name" which contains the results from the simulation
    '''

    file_ = open(f'{name}.pck', 'wb') # wb means opening a new file in the folder of the simulation
    pickle.dump(results, file_)
    file_.close()


def open_file_(name_file):
    '''
    Function which opens a file saved using pickle.
    Input:
        name_file - the name of the file with the simulation results. The input must be a string 'anythin'
        i - index of the file to open, i.e which cycle index. Type: int 
    Output:
        The simulation results decoded from the saved pickle file. Typically Adult data [table, biomass, Product] after each cycle.
    '''

    with open(f'{name_file}.pck','rb') as _:
        opening = pickle.load(_)
    return opening 


def seed_finder(Ns,name,survivor):
    success = []
    for i in range(Ns):
        shape_ = (open_file_(f'{name}{i}')[0]).shape[0]
        if shape_ >= survivor:
            success.append(i)
    for i in success:
        print('Seed:',i)
        print(open_file_(f'{name}{i}')[0])
    return success
        
