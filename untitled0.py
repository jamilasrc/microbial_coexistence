# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:11:23 2023

@author: jamil
"""

############################### Packages ###################################

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import time
import pickle
from functools import reduce
import h5py

#############################################################################

##### testing map reduce with changing dataframe size #######################

df_to_write = pd.DataFrame(np.zeros((50,37)))
df_to_write.iloc[:,:-1] = np.random.rand(50,36)
df_to_write[df_to_write.columns[-1]] = np.tile(200,50)

df_to_write.to_csv("df.csv")

def add_rows(df):
    
    #df = df_to_write
    
    new_rows = pd.DataFrame(np.hstack((np.random.rand(5,36),np.array([np.tile(200,5)]).T)))
    new_df = pd.concat([df,new_rows],axis=0)
    
    return new_rows

def new_df(prev_chunk,new_chunk):
    
    return pd.concat([prev_chunk,new_chunk],axis=0)


chunks = pd.read_csv("df.csv",index_col=0)
func_to_chunk = map(add_rows,chunks)
final_df = reduce(new_df,func_to_chunk)

###########

chunks2 = pd.read_csv("df.csv",chunksize=10,index_col=0)

check_df = []

for chunk in chunks2:
    
    new_rows = pd.DataFrame(np.hstack((np.random.rand(5,36),np.array([np.tile(200,5)]).T)))
    chunk = pd.concat([chunk,new_rows])
    
check_df = pd.concat(check_df)
    
for chunk in chunks2:
    
    print(chunk)
        
    
###########################################

start_array = np.hstack((np.random.rand(50,36),np.array([np.tile(200,50)]).T))
f = h5py.File('testh5py.hdf5','w')
sa_dset = f.create_dataset('chunked',chunks=(10,37),data=start_array)   

check_h5 = []

for s in sa_dset.iter_chunks():
    
    new_rows = np.hstack((np.random.rand(5,36),np.array([np.tile(200,5)]).T))
    print(np.vsplit(sa_dset[s],2))
    check_h5.append(np.vstack((sa_dset[s],new_rows)))
    
    
sa_dset2 = f.create_dataset('resizeable',chunks=True,data=start_array,maxshape=(500,37))   

check_2 = []

for s2 in sa_dset2.iter_chunks():
    
    check_2.append(sa_dset2[s2])

# if it's resizeable, it will not do anything to unassigned rows

sa_dset2.resize((51,37))
              
sa_dset2[50,:] = np.random.rand(1,37)

check_2 = []

for s2 in sa_dset2.iter_chunks():
    
    check_2.append(sa_dset2[s2])

sa_dset3 = f.create_dataset('resizeable',chunks=True,data=np.random.randn(100,100),maxshape=(None,None))   

sa_dset3.resize((10**15,10**15))


###################

from numba import njit

@njit
def numba_h5(h5_obj_chunked):
    
    for s in h5_obj_chunked:
        
        a = np.multiply(s,2)


numba_h5(sa_dset2.iter_chunks)