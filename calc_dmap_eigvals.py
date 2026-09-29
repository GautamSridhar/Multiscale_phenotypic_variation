#data format library
import h5py
#numpy
import numpy as np
# import pandas as pd
import numpy.ma as ma
import sys
sys.path.append('./utils/')
import os
import copy
import partitioning_methods as cl
import coarse_graining as cgm
import operator_calculations as op_calc
import delay_embedding as embed
import stats
import time as tt
import scipy
import glob
import argparse
from scipy.sparse.linalg import eigsh

np.random.seed(42)

def diff_map_eigvals(D_int,sigma_mat):
    kern = np.exp(-D_int**2 / sigma_mat)
    qa = kern.sum(axis=0) ** -1
    Da = np.diag(qa)
    L = np.dot(Da, np.dot(kern, Da))
    M = np.dot(np.diag(np.sum(L, axis=1) ** -1), L)
    # M = op_calc.get_reversible_transition_matrix(M)
    
    diff_eig, _ =  np.linalg.eig(M)
    diff_eig[diff_eig<1e-12] = 1e-12
    sorted_indices = np.argsort(diff_eig.real)[::-1]
    diff_eig = diff_eig[sorted_indices].real
    return diff_eig[1:]

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help="Number of seeds to run",type=int,default=0)
    parser.add_argument('-dn','--DatasetName',help="Name of the dataset to analyze", default='filtered_phdata7_condition{}.h5',type=str)
    parser.add_argument('-out','--Out',help="path save",default='/flash/StephensU/Gautam/Multiscale_phenotypic_variation/Results/',type=str)
    args=parser.parse_args()


    s,k = np.array(np.loadtxt(args.Out+args.DatasetName + '/eigvals/iteration_indices.txt')[args.Idx],dtype=int) ## Update path here
    print('s={},k={}'.format(s,k),flush=True)

    D_int = np.load(args.Out + args.DatasetName + '/D_int.npy')
    print(D_int.shape,flush=True)
    n_points = int(0.50*len(D_int))
    k_sigma=k

    sorted_d = np.sort(D_int, axis=1)
    sigma = np.zeros(len(D_int))
    for i in range(len(D_int)):
        nonzero = sorted_d[i, sorted_d[i] > 0]
        if len(nonzero) >= k_sigma:
            sigma[i] = nonzero[k_sigma - 1]
        else:
            sigma[i] = nonzero[-1] if len(nonzero) > 0 else 1.0  # fallback
            
    sigma_mat = np.outer(sigma, sigma)
    np.random.seed(s)
    idx = np.random.choice(np.arange(0,len(D_int)),n_points,replace=False)
    eigs_shuffle = diff_map_eigvals(D_int[idx,:][:,idx],sigma_mat[idx,:][:,idx])

    save_path = args.Out + args.DatasetName + '/eigvals/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('Saving results',flush=True)

    f= h5py.File(args.Out + args.DatasetName +'/eigvals/diffmap_eigvals_s{}_k{}.h5'.format(s,k),'w')
    es_ = f.create_dataset('eigs_shuffle',eigs_shuffle.shape)
    es_[...] = eigs_shuffle
    s_ = f.create_dataset('seed',(1,))
    s_[...] = s
    k_ = f.create_dataset('K',(1,))
    k_[...] = k_sigma
    f.close()

if __name__ == "__main__":
    main(sys.argv)


