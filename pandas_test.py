# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:46:14 2023

@author: jamil
"""

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

import numba
from numba import njit

import Cython
#%load_ext Cython type in console


##############################################
def func_test(t,y):
    
    dy = np.zeros(len(y))
    
    for v_i in range(len(y)):
        
        dy[v_i] = 3*y[v_i]
    
    for i in range(3):
        
        dy += np.multiply(i,y)
        
    return dy

sol = solve_ivp(func_test, [0, 10], np.array([1])) # This works
####################################

# timing test for dataframe updating 
# loop over list of small tables
# loop + pd.concat over 1 large table
# no loop, df.sort_index().reset_index(drop=True) on one large table

list_df = []
large_df = np.zeros((50*100,3))
steps = np.append(np.arange(0,50*100,100),50*100)

for s in range(len(steps[:-1])):
    
    #s = steps[0] 
    #print(s)
    
    rand_df = random.rand(100,3)
    
    list_df.append(pd.DataFrame(rand_df))
    
    large_df[steps[s]:steps[s+1],:] = rand_df
    
large_df = pd.DataFrame(large_df)

large_df2 = large_df
large_df3 = large_df

# test speeds
rows_to_add = 50 # comparable for all methods

# loop over list of small tables
t_s1 = time.time()

rows_selected = np.random.randint(100,size=len(list_df))

rows_selected = np.sort(rows_selected)[::-1]

for t in range(len(list_df)):
    
    #t = 0
    
    row_s = rows_selected[t]
    
    df = list_df[t]
    
    new_row = df.iloc[np.array([row_s])] - 1
    new_row_pair = new_row + 0.5
    
    table1 = pd.concat([new_row,new_row_pair]).reset_index(drop = True)
    
    df = pd.concat([df.iloc[:row_s+1],table1,df.iloc[row_s+1:]]).reset_index(drop = True)

    list_df[t] = df

t_e1 = time.time()

# loop and concat
t_s2 = time.time()

rows_selected2 = np.random.randint(50*100,size=len(list_df))
rows_selected2 = np.sort(rows_selected2)[::-1]

for r in range(len(rows_selected2)):
    
    #r = 0
    
    row_s = rows_selected2[r]
    
    new_row = large_df.iloc[np.array([row_s])] - 1
    new_row_pair = new_row + 0.5
    
    table1 = pd.concat([new_row,new_row_pair]).reset_index(drop = True)

    large_df = pd.concat([large_df.iloc[:row_s+1],table1,large_df.iloc[row_s+1:]]).reset_index(drop = True)

t_e2 = time.time()

# no loop and concat
t_s3 = time.time()

rows_selected3 = np.random.randint(50*100,size=len(list_df))

rows_selected3 = np.array([0,1])
rows_selected3 = np.sort(rows_selected3)[::-1]

new_row = large_df2.iloc[rows_selected3] - 1
new_row_pair = new_row + 0.5
new_row.index = rows_selected3 + 0.05
new_row_pair.index = rows_selected3 + 0.1

large_df2 = pd.concat([large_df2,new_row,new_row_pair]).sort_index().reset_index(drop = True)

t_e3 = time.time()

print({'loop_list':(t_e1-t_s1),'loop_concat':(t_e2-t_s2),'noloop_concat':(t_e3-t_s3)})

actual_sparse = np.zeros((100,36))
actual_sparse[:,8] = 0.5
actual_sparse[:,-1] = 100
actual_sparse = pd.DataFrame(actual_sparse)
actual_sparse = actual_sparse.astype(pd.SparseDtype(float, fill_value=0))

manual_sparse = pd.DataFrame(np.tile(np.array([8,0.5,100]),(100,1)))

'real sparse : {:0.2f} bytes'.format(actual_sparse.memory_usage().sum() / 1e3)
'manual sparse : {:0.2f} bytes'.format(manual_sparse.memory_usage().sum() / 1e3)

####################################################################

#@njit
def test_pd(df):
    
    return df.to_numpy

test_df = pd.DataFrame(np.random.rand(100,100))
res = test_df.apply(test_pd,engine="numba",raw=True)

# Cpde below works
data = pd.Series(range(1_000_000))  # noqa: E225
roll = data.rolling(10)

def f(x):
    return np.sum(x) + 5
roll.apply(f, engine='numba', raw=True)

%%cython
def test_pd(df):
    
    return df.to_numpy

test_df = pd.DataFrame(np.random.rand(100,100))
res = test_df.apply(test_pd) # this works


import numpy as cnp

%%cython   
def cython_speed(mat_list):
        
    for mat in mat_array:
        
        mat * mat
    
@njit  
def numba_speed(mat_array):
  
    for mat in mat_array:
        
        np.multiply(mat,mat)
        
mat_rep = np.random.rand(1000,1000)

numba_speed([mat_rep,mat_rep,mat_rep,mat_rep,mat_rep])
t_sn = time.time()
numba_speed([mat_rep,mat_rep,mat_rep,mat_rep,mat_rep])
t_en = time.time()

cython_speed([mat_rep,mat_rep,mat_rep,mat_rep,mat_rep])
t_sc = time.time()
cython_speed([mat_rep,mat_rep,mat_rep,mat_rep,mat_rep])
t_ec = time.time()

print({"numba":(t_en-t_sn),"cython":(t_ec-t_sc)}



df = pd.DataFrame(np.random.rand(100))

@njit
def numba_test(a=df[0].values):
    
    np.multiply(a,2)

numba_test()

def numba_applied(col):
    
    return np.multiply(col,2)

df = pd.DataFrame(np.random.rand(1000)).rolling
df.apply(numba_applied,engine="numba",raw=True)


def sdf_to_dense(df):
    
    '''
    Input
        df_to_np - . Type: df.to_numpy or df['col'].values
    '''
    min_param = 0
    max_param = 50
    
    breakpoint()
    
    df_to_np = df.values
    #breakpoint()
    dense_mat = np.zeros((df_to_np.shape[0],(max_param-min_param)))
    
    df_to_np = df.values
    dense_mat = np.zeros((df_to_np.shape[0],(max_param-min_param)))
    
    dense_mat[np.arange(df_to_np.shape[0]),(df_to_np[:,0]-1).astype('int32')] = df_to_np[:,1]
    #dense_mat[ancest_r,:] = ancest_parms
    
    return dense_mat


df_test = pd.DataFrame(np.vstack((np.repeat(np.arange(1,51),2),np.tile([0.5,-0.5],50))).T)
df_test.apply(sdf_to_dense)

sdf_to_dense(df_test)

######

a = pd.DataFrame(np.array([1,2,3]))
b = pd.DataFrame(np.array([2,3,4]))

pd.concat([a,b],axis=1).T

a.concat([a,b])

######

a = pd.DataFrame(np.random.rand(50,3))
b = pd.DataFrame(np.random.rand(10,3))

indexer = [np.array([0,0,0,3,3,4,7,7,7,7]),
                    np.arange(1,b.shape[0]+1)]

index_a = [np.arange(a.shape[0]),
           np.arange(a.shape[0])]

b.index = indexer
a.index = index_a

c = pd.concat([a,b]).sort_index()
c.index = [np.arange(c.shape[0]),np.arange(c.shape[0])]

###################################

df = pd.DataFrame(np.random.rand(5,3))

df.index = np.array([[1,1,2,2,2],[0,1,2,3,4]])

df_gb = df.groupby(level=0)


# testing chunking algorithms

def csum(d_m):
    
    return np.cumsum(d_m,axis=0)

v_func = np.vectorize(csum)

#@njit
def test_calc(mat,vec):
    
    #breakpoint()
    
    return np.matmul(mat,vec)
    


# initialise dummy data

df_test = pd.DataFrame(np.vstack((np.random.randint(0,100,100),np.random.rand(100),np.random.randint(0,400,100))).T)

# intialise dummy ancestor index
ancest_ind1 = np.array([0,3,10,14,18,25,35,40,60,66,70,85,100,100]) # have to have beginning and end for groupby

# test 1 using pandas .group_by() = ancestor ind, implement over 
# I guess this doesn't require any actual chunking

def sdf_to_dense1(df,min_param,max_param,ancest_row):
    
    #breakpoint()
    
    dense_mat = np.zeros((df.shape[0],(max_param-min_param)))
    
    dense_mat[np.arange(df.shape[0]),df[0].astype('int32')] = df[1]

    return np.cumsum(dense_mat,axis=0)

def func_to_apply(chunk,a_i):
    
    #breakpoint()
    
    new_vec = np.array(df_test[2])
    
    new_mat = sdf_to_dense1(chunk,0,100,a_i)
    
    mat_calc = test_calc(new_mat,new_vec)
    
    return mat_calc, mat_calc
    
t_s1 = time.time()
df_gb = df_test.groupby(np.repeat(np.arange(len(ancest_ind1[:-1])),np.diff(ancest_ind1)))
%timeit df_gb.apply(func_to_apply,a_i=ancest_ind1) 
df_as = np.array_split(df_test,ancest_ind1[1:-2])
%timeit [func_to_apply(df,ancest_ind1) for df in df_as] # this is about x2 faster. If we do this, I feel like we may as well use arrays throughout

a,b = zip(*[func_to_apply(df,ancest_ind1) for df in df_as]) 


b = func_to_apply(df_gb,ancest_ind1)
t_e1 = time.time()


# test 2 using pandas .group_by() = chunk, then implement 

chunk_size = 15 # no. rows, not no. species

def sdf_to_dense2(df,min_param,max_param,ancest_row):
    
    breakpoint()
    
    dense_mat = np.zeros((df.shape[0],(max_param-min_param)))
    
    dense_mat[np.arange(df.shape[0]),df[0].astype('int32')] = df[1]

    # cumsum
    dense_mat = np.vsplit(dense_mat,ancest_row[1:])

    return np.concatenate(v_func(dense_mat),axis=0)

def func_to_apply(chunk):
    
    #breakpoint()
    
    new_vec = np.array(df_test[2])
    
    new_mat = sdf_to_dense2(chunk,0,100,ancest_ind1)
    
    mat_calc = test_calc(new_mat,new_vec)
    
    return mat_calc

##################################### 

# testing array_split
arry_test = np.array([[1,2,3],[4,5,6]])
split_ind = [1,2]
a,b = np.array_split(arry_test,split_ind)[:-1]

#######

@njit
def matmul_numba(mat,vec):
    return mat @ vec


mat = np.array([[1.0,2.0,3.0],[1.0,2.0,3.0]])
vec = np.array([2.0,3.0,4.0])
%timeit matmul_numba(mat,vec)
%timeit np.matmul(mat,vec)

from numba import vectorize, int32, int64, float32, float64

@vectorize
def csum_VecWithNumba(d_m):
    
    return np.cumsum(d_m,axis=0)

df_test = pd.DataFrame(np.vstack((np.random.randint(0,100,100),np.random.rand(100),np.random.randint(0,400,100))).T)
ancest_ind1 = np.array([0,3,10,14,18,25,35,40,60,66,70,85,100,100]) # have to have beginning and end for groupby
df_as = np.array_split(np.array(df_test),ancest_ind1[1:-2])

csum_VecWithNumba(df_as)





