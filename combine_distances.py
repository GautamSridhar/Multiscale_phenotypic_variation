#data format library
import h5py

#numpy
import numpy as np
import pandas as pd
import numpy.ma as ma
%matplotlib inline


import matplotlib
import matplotlib.pyplot as plt
# %matplotlib notebook
import sys
sys.path.append('./utils/')
import matplotlib.colors as pltcolors
import os
import copy
import partitioning_methods as cl
import operator_calculations as op_calc
import delay_embedding as embed
import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time as tt
import scipy
import glob


np.random.seed(42)
def main(argv):
    t_0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-dn','--DatasetName',help="Name of the dataset to analyze", default='filtered_phdata7_condition{}.h5',type=str)
    parser.add_argument('-out','--Out',help="path to save",default='/flash/StephensU/Gautam/Multiscale_phenotypic_variation/Results/',type=str)
    args=parser.parse_args()

    indices_all=[]
    path_to_filtered_data =  args.Out + args.DatasetName + '/'
    # path_to_filtered_data = '/flash/StephensU/Gautam/HMD/Results/bacteria_all2/'
    for k in range(100):
        print(k)
        indices = np.loadtxt(path_to_filtered_data+'/indices/all_indices_{}.txt'.format(k))
        indices_all.append(np.array(indices,dtype=int))
        # k+=1

    q_range = np.unique(np.array(np.logspace(0,2,50),dtype=int))[1:]
    # q_range = np.unique(np.array(np.logspace(0,2.3,75),dtype=int))[1:] ## Uncomment for zebrafish and bacteria results


    Dc_q_data = np.zeros((len(q_range),nphens,nphens))
    Dc_q_eps = np.zeros((len(q_range),nphens,nphens))
    for idx in range(len(indices_all)):
        f= h5py.File(path_to_filtered_data+'/Eps_parallel/Ds_q_idx_{}.h5'.format(idx),'r')
        Ds_q_indices = np.array(f['Ds_q'],dtype=float)
        f.close()
        idx_i = indices_all[idx][:,0]
        idx_j = indices_all[idx][:,1]
        for k in range(len(indices_all[idx])):
            i,j = indices_all[idx][k]
            Dc_q_data[:,i,j] = Ds_q_indices[k,:,0]
            Dc_q_data[:,j,i] = Ds_q_indices[k,:,0]
            Dc_q_eps[:,i,j] = Ds_q_indices[k,:,1]
            Dc_q_eps[:,j,i] = Ds_q_indices[k,:,1]

        print(idx)

    print(path_to_filtered_data)
    f = h5py.File(path_to_filtered_data+'distances_combined.h5','w')
    Dd = f.create_dataset('Dc_q_data',Dc_q_data.shape)
    Dd[...] = Dc_q_data
    De = f.create_dataset('Dc_q_eps',Dc_q_eps.shape)
    De[...] = Dc_q_eps
    q_range_ = f.create_dataset('q_range',q_range.shape)
    q_range_[...] = q_range
    f.close()

    t_f = time.time()
    print('It took {:.2f} minutes.'.format((t_f-t_0)/60.))
    
if __name__ == "__main__":
    main(sys.argv)