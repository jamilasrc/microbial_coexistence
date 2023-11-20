# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:52:55 2023

@author: jamil
"""

from simulation_functions_jamila_2 import *

import numpy as np
import pandas as pd

########### initialise subpopulations dataframe ############

TID = 10**4 # total initial cell density
Nc = 50 # number of species

nCell0 = np.repeat(TID/Nc,Nc)

label = ['Parameter', 'Change', 'Number']
initial_table = pd.DataFrame(np.vstack((np.zeros(Nc),np.zeros(Nc),nCell0)).T,columns=label)
initial_table = initial_table.astype({'Parameter':'int32'})
initial_table.index = [np.arange(initial_table.shape[0]).astype('int32'),np.arange(initial_table.shape[0]).astype('int32')]

ancest_row_init = np.arange(initial_table.shape[0]).astype('int32')


##################### initialise chemical concentrations #####################

# initial state: chemicals
Nm = 7
cMed_init = np.zeros(Nm)

############## initialise random parameters ####################

# initial and final time with timestep 
tau0 = 0
tauf = 250
T = 22 # maturation time to reach dilution threshold (21.75 last, 23.12 largest)
dtau = 0.01
 
kSat = 1e4 # % interaction strength saturation level of each population %why saturation  -phrasing
ExtTh = 0.1 # % population extinction threshold %high threshold to maintain (try lower); try number of individuals. (not part of pop)
#   Jamila: Extinction threshold is ridiculously high. Is this from Babak?
DilTh = 1e7 # % coculture dilution threshold

nGen = 1000; # number of generations 
GenPerRound = np.log(DilTh/TID)/np.log(2)
#Nr = round(nGen/GenPerRound); #% number of rounds of propagation
Nr = 1

############ Parameters involved in simulating populations dynamics ###############

#   These need to be split up into parameters allowed to evolve, and parameters that are fixed.
#   These parameters will be stored in different data structures for ease.

# Parameters Allowed to Evolve

# locally simulated parameters, where seed 1343 achieves species coexistence 
params_2075 = open_file_('params_coex_2075')
params_2075['alpha'] = params_2075['alpha'].T
params_2075['beta'] = params_2075['beta'].T

parms_to_evolve = np.hstack((params_2075['r0'],
                       params_2075['K'],
                       params_2075['alpha'],
                       params_2075['beta'],
                       params_2075['rho_plus'],
                       params_2075['rho_minus']))

# Make parameter index for dynamics calculation using Numba 
keys = ['r0','K','alpha','beta','rho_plus','rho_minus']
e_parm_ind = np.cumsum(np.array([params_2075.get(key).shape[1] for key in keys])).astype('int32')

# Fixed Parameters
#   given as an np.array and parm ind again.

fixed_parm = np.array([ExtTh,kSat])
f_parm_ind = np.array([0,1]).astype('int32')

# Evolution Parameters
mutation_p = 0.01

###################### Chunk_size ############################

chunk_size = 25

################### Simulation ###############################

nCell_final, cMed_final, biomass_table_final = dilution_evo(Nr, Nc, TID, tauf, dtau,
                                            initial_table, ancest_row_init,
                                            parms_to_evolve, e_parm_ind, fixed_parm, f_parm_ind, mutation_p,
                                            cMed_init,
                                            chunk_size,
                                            'name_')
