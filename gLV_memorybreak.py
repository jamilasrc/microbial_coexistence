# -*- coding: utf-8 -*-
"""
Lotka-Volterra model to break cluster
Created on Thu Oct 19 14:59:26 2023

@author: jamil
"""

import numpy as np
from numpy import random
import pandas as pd

from scipy.integrate import solve_ivp

import time

################

# gLV form: dS_i/dt = S_i*(g_i+sum(a_ij*S_j)) i = 1, ..., N
#   a_ij = 1 for i = j

###########################################################

# must start from high point        
def find_max_mem(start_size):
    
    current_dim = start_size
    
    up_bound = start_size
    low_bound = 0 
    
    while (up_bound-low_bound) > 100:
        
        try:
            
            #mat = np.random.rand(current_dim,current_dim)
            mat = np.ones((current_dim,current_dim))
            del mat
            
            new_dim = int((current_dim + up_bound)/2)
            low_bound = current_dim

        except MemoryError:
            
            print('Error: Memory requirements too high for',current_dim*current_dim,'parameters.')
            
            new_dim = int((current_dim + low_bound)/2)
            up_bound = current_dim
        
        current_dim = new_dim
        print(current_dim)
            
    #return current_dim * current_dim
    
##############################################

def gLV(current_dim): 
    
    # interaction terms
    mu_ri = 0 # mean
    sigma_ri = 0.5 # standard deviation
 
    # growth rates
    mu_g = 0 # mean
    sigma_g = 0.5 # standard deviation
    
    no_species = current_dim

    # Create random interaction matrix
    rand_intrct = sigma_ri*np.random.randn(no_species,no_species) + mu_ri

    np.fill_diagonal(rand_intrct, -1) # set a_ij = -1 for i = j to prevent divergence

    # Create random growth rates     
    rand_grwth = (abs(sigma_g*np.random.randn(no_species,1) + mu_g)).reshape((no_species,))

    def dSdt(t,spec):
        
        dS = np.multiply(rand_grwth + np.matmul(rand_intrct,spec), spec)
        
        return dS

    spec_0 = 20*np.ones((no_species,)) # set all initial species populations to 20 for now.
    t_end = 30

    result = solve_ivp(fun=dSdt,t_span=[0,t_end],rtol=10e-4,y0=spec_0)
    
    return result
    
    
def max_mem_model(start_size,function):
    
    current_dim = start_size
    
    up_bound = start_size
    low_bound = 0 
    
    while (up_bound-low_bound) > 100:
        
        try:
            
            t_s1 = time.time()
            model_res = function(current_dim)
            t_e1 = time(time)
            del model_res
            
            new_dim = int((current_dim + up_bound)/2)
            low_bound = current_dim
            
            runtime = (t_e1-t_s1)
            # we probably want to include some function that estimates memory usage. Of what? Just the matrix or?

        except MemoryError:
            
            print('Error: Memory requirements too high for',current_dim*current_dim,'parameters.')
            
            new_dim = int((current_dim + low_bound)/2)
            up_bound = current_dim
        
        current_dim = new_dim
        #print(current_dim)
        
    #return [current_dim,low_bound,up_bound,runtime]


max_memory = max_mem_model(100000,gLV)


def max_mem_chunkedmodel(start_size):
    
    current_dim = start_size
    
    up_bound = start_size
    low_bound = 0 
    
    while (up_bound-low_bound) > 100:
        
        try:
            
            as_array = np.ones((current_dim,3))
            as_df = pd.DataFrame(as_array)

            del as_array
            del as_df
            
            new_dim = int((current_dim + up_bound)/2)
            low_bound = current_dim

        except MemoryError:
            
            print('Error: Memory requirements too high for generating an array and dataframe of',current_dim,' by 3 dimensions.')
            
            new_dim = int((current_dim + low_bound)/2)
            up_bound = current_dim
        
        current_dim = new_dim
        print(current_dim)
        
    return current_dim

max_mem = s








