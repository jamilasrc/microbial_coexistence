# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:38:15 2023

@author: jamil
"""

############################### Packages ###################################

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import time
import pickle
import h5py

## DEPRECATED
#from numba import njit
#from numba.typed import List

#############################################################################


############################### Functions ###################################

########## Evolution ##################

def mutation_df(biomass_table,min_param,max_param,mutation_p,ancest_row):
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
    
    num_mutation = mutation_freq(mutation_p,np.round(biomass_table['Number']).astype('int32'),biomass_table.shape[0]) # calculate no. new mutants
    #   per ancestral subpopulation.
    
    biomass_table['Number'] -= num_mutation # remove new mutants from ancestral subpopulation
    ancest_row[1:] += np.cumsum(np.multiply(num_mutation[:-1],2)) # update ancest_row with new position of species ancestors in the new table.
    #   (shouldn't need to return , but check)
    
    #mutants_to_add, final_mutation = create_mutants_np(min_param,max_param,mutation_distribution,num_mutation)
    mutants_to_add = create_mutants_np(min_param,max_param,mutation_distribution_n,num_mutation) # generate dataframe of new mutants
    
    new_table = pd.concat([biomass_table,mutants_to_add]).sort_index() # add new_mutants to big_table, inserted into their correct place using indexing
    new_table.index = [np.arange(new_table.shape[0]),np.arange(new_table.shape[0])] # update new_table index to standard multindexing.
    
    return new_table

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
    #breakpoint()
    
    no_mutants = np.sum(num_mutation) # total no. mutants from no. mutants per ancestral subpopulation
    
    select_parameter = np.random.randint(min_param,max_param,no_mutants) # vectorised selection of parameters to mutate in the new mutants
    mutation_change = mutation_distribution_n(no_mutants) # vectorised calculation of change in selected parameters in mutants
    
    mutant_dat = np.repeat(np.vstack((select_parameter,mutation_change)),2,axis=1) # stack corresponding parameter and change 
    #   per mutant together, but duplicate each row so we can calculate the closeout rows.
    
    mutant_dat[1,1::2] = -mutant_dat[1,1::2] # closeout row ['Change'] = - mutant row ['Change']
    #mutant_dat = np.vstack((mutant_dat,np.tile(np.array([1,0]),int(mutant_dat.shape[1]/2)))).T
    
    mutant_dat = np.vstack((mutant_dat,np.tile(np.array([1,0]),sum(num_mutation)))).T # add new column for ['Number']. New mutant rows
    #   are filled with 1s, because one new mutant is generated from the ancestral subpop. Closeout rows are filled with 0s.
        
    
    indexer = [np.repeat(np.arange(len(num_mutation)),num_mutation*2),
               np.arange(1,mutant_dat.shape[0]+1)] # indexing that allows mutant and closeout rows to be inserted in the correct
    #   place. First index = index on ancestral subpop. 2nd index = standard indexing so after the new rows are inserted in the
    #   correct place during pd.concat(), rows remain in the correct order.
    mutant_df = pd.DataFrame(mutant_dat,columns=["Parameter","Change","Number"],
                             index=indexer) # convert np.array to dataframe
    mutant_df = mutant_df.astype({'Parameter':'int32'})
    
    return mutant_df

def mutation_distribution_n(no_mutants):
    return np.round(np.random.normal(size=no_mutants),2)

########### Chunking ##############

def closest_argmin(B,A):# modified from stackoverflow
    
    L = B.size
    sorted_idx = np.searchsorted(B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - B[sorted_idx-1]) < np.abs(A - B[sorted_idx])) )
    
    return np.unique(B[sorted_idx-mask])

######### Sparse-Dense Conversions ###########

def convrt_dyn_nochunk(biomass_table,min_param,max_param,ancest_row,parms_to_evolve,e_parm_ind):
    
    dense_matrix = sdf_to_dense(biomass_table,min_param,max_param,ancest_row,parms_to_evolve)
    
    return np.array_split(dense_matrix,e_parm_ind,axis=1)[:-1]


def sdf_to_dense(df,min_param,max_param,a_row,parms_to_evolve):
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
    
    #breakpoint()
    
    dense_mat = np.zeros((df.shape[0],(max_param-min_param)))
    
    dense_mat[np.arange(df.shape[0]),df['Parameter']] = df['Change']
    dense_mat[a_row,:] = parms_to_evolve
    
    # cumsum
    dense_mat = np.vsplit(dense_mat,a_row[1:])
    dense_mat = np.vstack([np.cumsum(d_m,axis=0) for d_m in dense_mat])
    
    return dense_mat

## DEPRECATED
def csum(dense_mat):
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
    
    return np.cumsum(dense_mat)

v_func = np.vectorize(csum) # above sum is vectorosed. This is the function implemented in sparse-to-dense array conversion.

##

 
##################################################

def cnvrt_dnmcs_new(t,spec,chem,
                      biomass_table,a_r_chunked,
                      parms_to_evolve,e_parm_ind,fixed_parm,f_parm_ind,min_param,max_param):
    
    dS_all = np.zeros(spec.shape)
    dC_all = np.zeros(chem.shape)
    
    for i in range(len(a_r_chunked)-1):
        
        #breakpoint()
        
        chunk = a_r_chunked[i]
        end_of_chunk = a_r_chunked[i+1][0]
        
        #breakpoint()
        
        #initiliase chunked data
        s_chunk = spec[chunk[0]:end_of_chunk,:]
        df_chunk = biomass_table.iloc[chunk[0]:end_of_chunk,:]
        ancest_parm_chunk = parms_to_evolve[i]
        
        
        dense_matrix = sdf_to_dense2(df_chunk,min_param,max_param,chunk,ancest_parm_chunk)
        
        #breakpoint()
        
        dS_chunk, dC_chunk = dynamics_calc_chunk(t,
                                           s_chunk,dense_matrix,e_parm_ind,fixed_parm,f_parm_ind,
                                           chem)
        
        dS_all[chunk[0]:end_of_chunk,:] = dS_chunk
        dC_all += dC_chunk
        
        
    return np.concatenate((dS_all,dC_all))

def sdf_to_dense2(df,min_param,max_param,chunk,ancest_parm_chunk):
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
    
    dense_mat = np.zeros((df.shape[0],(max_param-min_param)))
    
    dense_mat[np.arange(df.shape[0]),df['Parameter']] = df['Change']
    dense_mat[(chunk-chunk[0]),:] = ancest_parm_chunk
    
    dense_mat = np.vsplit(dense_mat,(chunk[1:]-chunk[0]))
    dense_mat = np.vstack([np.cumsum(d_m,axis=0) for d_m in dense_mat])

    return dense_mat

def create_sparsedf(biomass_table,ancest_row,parms_to_evolve,min_param,max_param):
    
    #breakpoint()
    
    sparse_df = np.zeros((biomass_table.shape[0],(max_param-min_param)))
    
    sparse_df[np.arange(biomass_table.shape[0]),biomass_table['Parameter']] = biomass_table['Change']
    sparse_df[ancest_row,:] = parms_to_evolve
    
    sparse_df = pd.DataFrame(sparse_df)
    sparse_df = sparse_df.astype(pd.SparseDtype("float",fill_value=0.0))

    return sparse_df

################### Chunking with HDF5 Files ####################################


def prepare_h5(biomass_table,min_param,max_param,ancest_row,parms_to_evolve,chunk_size):
    
    #breakpoint()
    
    # initialise hdf5 file - don't dp tjhis
    file_start = h5py.File('modelh5py.hdf5','w')
    
    # convert compressed data to dense form
    dense_matrix = sdf_to_dense(biomass_table,min_param,max_param,ancest_row,parms_to_evolve)
    dm_indexed = np.hstack((np.arange(dense_matrix.shape[0]).reshape((dense_matrix.shape[0],1)),
                            dense_matrix))
    
    # create h5py dataset
    model_dset = file_start.create_dataset('unlimited',chunks=(chunk_size,dm_indexed.shape[1]),
                                           data=dm_indexed,maxshape=(None,dm_indexed.shape[1]))   
  
    return model_dset

######### Dynamics Calculations ########

def dynamics_calc_chunk(t,
                        s_chunk,dense_matrix,e_parm_ind,fixed_parm,f_parm_ind,
                        chem):
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
    
    #breakpoint()
    
    # What will be the tidy version
    r0, K, alpha, beta, rho_plus, rho_minus = np.array_split(dense_matrix,e_parm_ind,axis=1)[:-1]
    kSat = fixed_parm[f_parm_ind[1]]
     
    dS, dC = interaction_chunk(s_chunk,chem,r0,K,alpha,beta,rho_plus,rho_minus,kSat)
    
    return dS, dC

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
    rho_net = (rho_pos + rho_min) @ chem
    dS = (r0 + rho_net) * s_chunk
    
    Ce =  np.ones(len(s_chunk)) * chem
    A = np.reciprocal(Ce + kSat) * Ce
    dC = s_chunk.T @ (beta - (alpha * A.T))
    
    return dS, dC.T


######### File Saving #################

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


######### Simulations #################


def dilution_evo(Nr,Nc,TID,tauf,dtau,
                 initial_table,ancest_row_init,
                 parms_to_evolve,e_parm_ind,fixed_parm,f_parm_ind,mutation_p,
                 cMed_init,
                 chunk_size,
                 name_):
    
    '''
    Highest function in hierachy.
    Simulation of dilution rounds, when total biomass reaches dilution threshold, the next round starts
    Input:
    '''
    
    # initial variable assignment 
    cMed_m = cMed_init
    
    cellRatio = np.repeat(1/Nc,Nc)
    nCell_m = TID*cellRatio
    
    biomass_table_m = initial_table
    ancest_row_m = ancest_row_init
    
    for i in range(Nr):
        
        # dilution of chemicals and species numbers
        cMed_m *= TID/sum(nCell_m)  # dilute the concentration of chemicals accourding to the new size of total species number, 
        
        cellRatio = np.divide(nCell_m,np.sum(nCell_m)) # new cell ratio, where we dilute after the total number of cells is bigger than dilution threshold
        nCell_m = TID*cellRatio
        biomass_table_m['Number'] = nCell_m
        
        # simulation
        cMed_m, nCell_m, biomass_table_m, ancest_row_m = maturation_evo(tauf,dtau,
                           biomass_table_m,ancest_row_m,
                           parms_to_evolve,e_parm_ind,fixed_parm,f_parm_ind,mutation_p,
                           cMed_m,
                           chunk_size)

        # save species dynamics (do we not want to save chemical concentrations?)
        result_save_(f'{name_}_{i}',nCell_m)
        
        #breakpoint()
        
    return nCell_m, cMed_m, biomass_table_m, ancest_row_m
    

# NOTE: This function will be very slow because this for loop. I think we need to increase the timestep aka dtau
# I am also a bit concerned about the number of duplicate dataframes created in memory, especially with such a
#   small timestep.
def maturation_evo(tauf,dtau,
                   biomass_table_m,ancest_row_m,
                   parms_to_evolve,e_parm_ind,fixed_parm,f_parm_ind,mutation_p,
                   cMed_m,
                   chunk_size):
    '''
    2nd highest function in hierachy, called by dilution_evo()

    Parameters
    ----------
    tauf : TYPE
        DESCRIPTION.
    dtau : TYPE
        DESCRIPTION.
    biomass_table_m : TYPE
        DESCRIPTION.
    ancest_row_m : TYPE
        DESCRIPTION.               
    parms_to_evolve : TYPE
        DESCRIPTION.
    e_parm_ind : TYPE
        DESCRIPTION.
    fixed_parm : TYPE
        DESCRIPTION.
    f_parm_ind : TYPE
        DESCRIPTION.
    mutation_p : TYPE
        DESCRIPTION.
    cMed_m : TYPE
        DESCRIPTION.

    Returns
    -------
    cMed : TYPE
        DESCRIPTION.
    nCell : TYPE
        DESCRIPTION.
    biomass_table : TYPE
        DESCRIPTION.
        

    '''
    
    # NOTE: This function will be very slow because this for loop. I think we need to increase the timestep aka dtau
    # I am also a bit concerned about the number of duplicate dataframes created in memory, especially with such a
    #   small timestep.

    
    step = int(tauf/dtau)
    
    cMed = cMed_m
    biomass_table = biomass_table_m
    ancest_row = ancest_row_m

    
    for i in range(step):
         
        cMed, nCell, biomass_table, ancest_row = species_dynamics_evo_bnchmrk(dtau,
                                                     biomass_table,ancest_row,
                                                     parms_to_evolve,e_parm_ind,
                                                     fixed_parm,f_parm_ind,mutation_p,
                                                     cMed,
                                                     chunk_size)
    
    return cMed, nCell, biomass_table, ancest_row


def species_dynamics_evo(dtau,
                         biomass_table,ancest_row,
                         parms_to_evolve,e_parm_ind,fixed_parm,f_parm_ind,mutation_p,
                         cMed,
                         chunk_size):
    '''
    This function runs evolution and dynamics. Called by mutation_evo()

    Parameters
    ----------
    dtau : TYPE
        DESCRIPTION.
    biomass_table : TYPE
        DESCRIPTION.
    ancest_row : TYPE
        DESCRIPTION.
    parms_to_evolve : TYPE
        DESCRIPTION.
    e_parm_ind : TYPE
        DESCRIPTION.
    fixed_parm : TYPE
        DESCRIPTION.
    f_parm_ind : TYPE
        DESCRIPTION.
    mutation_p : TYPE
        DESCRIPTION.
    cMed : TYPE
        DESCRIPTION.

    Returns
    -------
    dSCdt : TYPE
        DESCRIPTION.

    '''
    
    ############### Extract revelent info ###########
    
    min_param = 0
    max_param = parms_to_evolve.shape[1]
    
    ################ Implement evolution ##################
    
    biomass_table = mutation_df(biomass_table,min_param,max_param,mutation_p,ancest_row) 
    # biomass_table returned, ancest_row updated within.
    
    ############# Dyanmics calculations + sparse-dense conversions #############
    
    spec_len = biomass_table.shape[0]
    
    pte_placeholder = parms_to_evolve
    
    #breakpoint()
    
     
    # Chunked dynamics calculations
    if (ancest_row >= chunk_size).any():
        
            #breakpoint()
            
        chunk_rng = np.arange(0,ancest_row[-1],chunk_size)
        
        chunk_ind = closest_argmin(ancest_row, chunk_rng)
        a_r_chunked = np.array_split(ancest_row,
                                     (np.nonzero(np.in1d(ancest_row,chunk_ind))[0])[1:])
        a_r_chunked.append(np.array([biomass_table.shape[0]]))
        
        parms_to_evolve = np.vsplit(parms_to_evolve,
                                         (np.nonzero(np.in1d(ancest_row,chunk_ind))[0])[1:])
        

        dynamics_sol = solve_ivp(fun=dDynamics_dt_chunk, t_span=[0,dtau], method = 'LSODA',
                               rtol = 10e-9, y0 = np.concatenate((biomass_table['Number'].values,cMed)),
                               args = (spec_len,biomass_table,a_r_chunked,parms_to_evolve,
                                       e_parm_ind,fixed_parm,f_parm_ind,min_param,max_param))
           
        #breakpoint()
          
        # No chunking   
    else: 
        
        parms_to_evolve = pte_placeholder
        
        r0, K, alpha, beta, rho_plus, rho_minus = convrt_dyn_nochunk(biomass_table,min_param,max_param,ancest_row,parms_to_evolve,e_parm_ind)
        kSat = fixed_parm[f_parm_ind[1]]
        
        dynamics_sol = solve_ivp(fun=dDynamics_dt, t_span=[0,dtau], method = 'LSODA',
                             rtol = 10e-9, y0 = np.concatenate((biomass_table['Number'],cMed)),
                             args = (spec_len, 
                                     r0, K, alpha, beta, rho_plus, rho_minus, kSat))
        
            #breakpoint()
         
    ##################### Update population dynamics and chemical mediator conc.  ##################
    
    breakpoint()
    
    nCell = dynamics_sol.y[:biomass_table.shape[0],-1].T
    ExtTh = fixed_parm[f_parm_ind[0]]
    nCell[nCell<ExtTh] = 0
    
    biomass_table['Number'] = nCell
    
    cMed = dynamics_sol.y[biomass_table.shape[0]:,-1]
    
    return cMed, nCell, biomass_table, ancest_row


def species_dynamics_evo_bnchmrk(dtau,
                         biomass_table,ancest_row,
                         parms_to_evolve,e_parm_ind,fixed_parm,f_parm_ind,mutation_p,
                         cMed,
                         chunk_size):
    '''
    This function runs evolution and dynamics. Called by mutation_evo()

    Parameters
    ----------
    dtau : TYPE
        DESCRIPTION.
    biomass_table : TYPE
        DESCRIPTION.
    ancest_row : TYPE
        DESCRIPTION.
    parms_to_evolve : TYPE
        DESCRIPTION.
    e_parm_ind : TYPE
        DESCRIPTION.
    fixed_parm : TYPE
        DESCRIPTION.
    f_parm_ind : TYPE
        DESCRIPTION.
    mutation_p : TYPE
        DESCRIPTION.
    cMed : TYPE
        DESCRIPTION.

    Returns
    -------
    dSCdt : TYPE
        DESCRIPTION.

    '''
    
    ############### Extract revelent info ###########
    
    min_param = 0
    max_param = parms_to_evolve.shape[1]
    
    ################ Implement evolution ##################
    
    biomass_table = mutation_df(biomass_table,min_param,max_param,mutation_p,ancest_row) 
    # biomass_table returned, ancest_row updated within.
    
    ############# Dyanmics calculations + sparse-dense conversions #############
    
    spec_len = biomass_table.shape[0]
    
    pte_placeholder = parms_to_evolve
     
    ################## Chunked dynamics , no numba #################
    #if (ancest_row >= chunk_size).any():

    chunk_rng = np.arange(0,ancest_row[-1],chunk_size)
    
    
    chunk_ind = closest_argmin(ancest_row, chunk_rng)
    a_r_chunked = np.array_split(ancest_row,
                                 (np.nonzero(np.in1d(ancest_row,chunk_ind))[0])[1:])
    a_r_chunked.append(np.array([biomass_table.shape[0]]))
    
    parms_to_evolve = np.vsplit(parms_to_evolve,
                                     (np.nonzero(np.in1d(ancest_row,chunk_ind))[0])[1:])
    
    time_reps = np.zeros((200,))
    
    for rep in range(200):
        
        t1 = time.time()
    
  
        dynamics_sol1 = solve_ivp(fun=dDynamics_dt_chunk, t_span=[0,dtau], method = 'LSODA',
                               rtol = 10e-9, y0 = np.concatenate((biomass_table['Number'].values,cMed)),
                               args = (spec_len,biomass_table,a_r_chunked,parms_to_evolve,
                                       e_parm_ind,fixed_parm,f_parm_ind,min_param,max_param))
        t2 = time.time()
        
        time_reps[rep] = t2-t1
        
    ############### Chunking with pandas sparse dataframe ###########
    
    parms_to_evolve = pte_placeholder
    
    sparse_df = create_sparsedf(biomass_table,ancest_row,parms_to_evolve,min_param,max_param)
    
    time_reps2 = np.zeros((200,))
    
    for rep in range(200):
        
        t3 = time.time()
        
        dynamics_sol2 = solve_ivp(fun=dDynamics_sparse, t_span=[0,dtau], method = 'LSODA',
                               rtol = 10e-9, y0 = np.concatenate((biomass_table['Number'].values,cMed)),
                               args = (spec_len,sparse_df,a_r_chunked,e_parm_ind,fixed_parm,f_parm_ind))
        
        t4 = time.time()
        
        time_reps2[rep] = t4-t3
        
    
    ######################### Chunking with h5py #######################
    
    parms_to_evolve = pte_placeholder
    
    model_dset = prepare_h5(biomass_table,min_param,max_param,ancest_row,parms_to_evolve,chunk_size)
    
    time_reps3 = np.zeros((200,))
    
    for rep in range(200):
        
        t5 = time.time()
  
        dynamics_sol3 = solve_ivp(fun=dDynamics_h5, t_span=[0,dtau], method = 'LSODA',
                               rtol = 10e-9, y0 = np.concatenate((biomass_table['Number'].values,cMed)),
                               args = (spec_len,model_dset,e_parm_ind,fixed_parm,f_parm_ind))
        
        t6 = time.time()
        
        time_reps3[rep] = t6-t5
        
    ####################### No chunking ###########################################
    
    parms_to_evolve = pte_placeholder
    
    r0, K, alpha, beta, rho_plus, rho_minus = convrt_dyn_nochunk(biomass_table,min_param,max_param,ancest_row,parms_to_evolve,e_parm_ind)
    kSat = fixed_parm[f_parm_ind[1]]
    
    time_reps4 = np.zeros((200,))
    
    for rep in range(200):
        
        t7 = time.time()
    
        dynamics_sol4 = solve_ivp(fun=dDynamics_dt, t_span=[0,dtau], method = 'LSODA',
                             rtol = 10e-9, y0 = np.concatenate((biomass_table['Number'],cMed)),
                             args = (spec_len, 
                                     r0, K, alpha, beta, rho_plus, rho_minus, kSat))
        t8 = time.time()
        
        time_reps4[rep] = t8-t7
       
        #breakpoint()
         
    ##################### Update population dynamics and chemical mediator conc.  ##################
    
    b1 = mean_sterror(time_reps)
    b2 = mean_sterror(time_reps2)
    b3 = mean_sterror(time_reps3)
    b4 = mean_sterror(time_reps4)
    
    breakpoint()
    
    nCell = dynamics_sol.y[:biomass_table.shape[0],-1].T
    ExtTh = fixed_parm[f_parm_ind[0]]
    nCell[nCell<ExtTh] = 0
    
    biomass_table['Number'] = nCell
    
    cMed = dynamics_sol.y[biomass_table.shape[0]:,-1]
    
    return cMed, nCell, biomass_table, ancest_row




######################### dYdt functions ####################################################

def dDynamics_dt(t,y,
                 spec_len,
                 r0, K, alpha, beta, rho_plus, rho_minus, kSat):
    
    spec = y[:spec_len]
    spec = np.reshape(spec,(spec.shape[0],1))
    
    chem = y[spec_len:]
    chem = np.reshape(chem,(chem.shape[0],1))
    
    dSCdt = np.concatenate((interaction_chunk(spec,chem,r0,K,alpha,beta,rho_plus,rho_minus,kSat)))
    dSCdt = np.reshape(dSCdt,(dSCdt.shape[0],))
    
    return dSCdt


def dDynamics_dt_chunk(t,y,
                       spec_len,
                       biomass_table,a_r_chunked,
                       parms_to_evolve,e_parm_ind,fixed_parm,f_parm_ind,min_param,max_param):
    
    spec = y[:spec_len]
    spec = np.reshape(spec,(spec.shape[0],1))
    
    chem = y[spec_len:]
    chem = np.reshape(chem,(chem.shape[0],1))
    
    dSCdt = cnvrt_dnmcs_new(t,spec,chem,
                          biomass_table,a_r_chunked,
                          parms_to_evolve,e_parm_ind,fixed_parm,f_parm_ind,min_param,max_param)
    dSCdt = np.reshape(dSCdt,(dSCdt.shape[0],))
    
    #breakpoint()
    
    return dSCdt

def dDynamics_h5(t,y,
                 spec_len,
                 h5_dset,
                 e_parm_ind,fixed_parm,f_parm_ind):
    
    
    spec = y[:spec_len]
    spec = np.reshape(spec,(spec.shape[0],1))
    
    chem = y[spec_len:]
    chem = np.reshape(chem,(chem.shape[0],1))
    
    dS_all = np.zeros(spec.shape)
    dC_all = np.zeros(chem.shape)
    
    for chunk_i in h5_dset.iter_chunks():
        
        chunk = h5_dset[chunk_i]

        s_chunk = spec[chunk[:,0].astype('int64')]
        
        dS_chunk, dC_chunk = dynamics_calc_chunk(t,
                                           s_chunk,chunk[:,1:],e_parm_ind,fixed_parm,f_parm_ind,
                                           chem)
        
        dS_all[chunk[:,0].astype('int64'),:] = dS_chunk
        dC_all += dC_chunk
        
    dSCdt = np.concatenate((dS_all,dC_all))
    dSCdt = np.reshape(dSCdt,(dSCdt.shape[0],))
    
    #breakpoint()
    
    return dSCdt


def dDynamics_sparse(t,y,
                       spec_len,
                       sparse_df,a_r_chunked,
                       e_parm_ind,fixed_parm,f_parm_ind):
    
    spec = y[:spec_len]
    spec = np.reshape(spec,(spec.shape[0],1))
    
    chem = y[spec_len:]
    chem = np.reshape(chem,(chem.shape[0],1))
    
    dS_all = np.zeros(spec.shape)
    dC_all = np.zeros(chem.shape)
    
    for i in range(len(a_r_chunked)-1):
        
        #breakpoint()
        
        chunk = a_r_chunked[i]
        end_of_chunk = a_r_chunked[i+1][0]
        
        #initiliase chunked data
        s_chunk = spec[chunk[0]:end_of_chunk,:]
        df_chunk = sparse_df.iloc[chunk[0]:end_of_chunk,:]
        
        dense_matrix = np.vsplit(np.asarray(df_chunk),(chunk[1:]-chunk[0]))
        dense_matrix = np.vstack([np.cumsum(d_m,axis=0) for d_m in dense_matrix])

        dS_chunk, dC_chunk = dynamics_calc_chunk(t,
                                           s_chunk,dense_matrix,e_parm_ind,fixed_parm,f_parm_ind,
                                           chem)
        
        dS_all[chunk[0]:end_of_chunk,:] = dS_chunk
        dC_all += dC_chunk
        
    dSCdt = np.concatenate((dS_all,dC_all))
    dSCdt = np.reshape(dSCdt,(dSCdt.shape[0],))
    
    #breakpoint()
    
    return dSCdt

####################### benchmarking ###################################

def mean_sterror(time_array):
    
    mean = np.mean(time_array)
    sterror = np.std(time_array,ddof=1)/np.sqrt(np.size(time_array))
    
    return mean, sterror

