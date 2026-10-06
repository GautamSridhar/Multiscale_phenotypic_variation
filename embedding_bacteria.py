#data format library
import argparse
import copy
import os
import sys
import functools
import concurrent.futures
sys.path.append('./utils/') ## Update Path here
import h5py
import glob
import re
#numpy
import numpy as np
import pandas as pd
import numpy.ma as ma
import partitioning_methods as cl
import delay_embedding as embed
import operator_calculations as op_calc
import stats
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import savgol_filter

import time

np.random.seed(42)


def unwrapma(x):
    # Adapted from numpy unwrap, this version ignores missing data
    idx = ma.array(np.arange(0,x.shape[0]), mask=x.mask)
    idxc = idx.compressed()
    xc = x.compressed()
    dd = np.diff(xc)
    ddmod = np.mod(dd+np.pi, 2*np.pi)-np.pi
    ddmod[(ddmod==-np.pi) & (dd > 0)] = np.pi
    phc_correct = ddmod - dd
    phc_correct[np.abs(dd)<np.pi] = 0
    ph_correct = np.zeros(x.shape)
    ph_correct[idxc[1:]] = phc_correct
    up = x + ph_correct.cumsum()
    return up


def calc_entropy_from_data(feats, sample_size,lengths_all,params):
    """
    Calculate the  for a given seed, delay K and cluster size.
    Parameters:
        data: datasets to perform the calculations on
        bouts: number of bouts to sample from each condition
        params: list containing seed, delay K and number of cluster N_cluster
    Returns:
            list with value of seed, delay, number of cluster and entropies
    """

    seed = params[0]
    K = params[1]
    N_cluster = params[2]

    min_count = 200
    print('Seed:{}'.format(seed))
    np.random.seed(seed)

    idx = np.random.choice(len(lengths_all), sample_size, replace=False)
    feats_ = ma.concatenate(feats[idx], axis=0)

    feats_[:,0] = (feats_[:,0] - ma.mean(feats_[:,0]))/ma.std(feats_[:,0])
    feats_[:,1] = (feats_[:,1] - ma.mean(feats_[:,1]))/ma.std(feats_[:,1])

    H = []
    #for kf,f0 in enumerate(np.arange(0,data_bootstrap.shape[0],div)):
    print('Starting delay {}, clusters {}:'.format(K, N_cluster), feats_.shape, flush=True)
    if ma.count(feats_,axis=0)[0]>min_count:
        traj_matrix = embed.trajectory_matrix(feats_,K=K-1)
        labels = cl.kmeans_knn_partition(traj_matrix,n_seeds=N_cluster, batchsize=5000)
        segments = op_calc.segment_maskedArray(labels)
        dtrajs = [labels[t0:tf] for t0,tf in segments]
        h = op_calc.get_entropy(dtrajs)
        H.append(h)
    print('Finished processing: delay {}, clusters {}'.format(K,N_cluster), traj_matrix.shape,flush=True)

    return [seed, K, N_cluster, H]


def main(argv):
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-seeds','--Seeds',help="Number of seeds to evaluate over",type=int,default=10)
    parser.add_argument('-recs','--Recs',help="Number of fish to sample from each morphotype",default=100,type=int)
    parser.add_argument('-dn','--DatasetName',help="Name of the dataset to save under", default='SF_CF_test',type=str)
    parser.add_argument('-out','--Out',help="path save",default='/flash/StephensU/Gautam/HMD/Results/',type=str)

    args=parser.parse_args()
    
    ## Load Data
    file_name = args.Out + args.DatasetName + '/bacteria_all_data.h5'

    f = h5py.File(file_name,'r')
    lengths_all = np.array(f['MetaData/lengths_data'],dtype=int)
    feats_all = ma.array(f['feats'],dtype=float)


    feats = []
    for i in range(len(lengths_all)):
        feats.append(feats_all[i,:lengths_all[i],:])
   
    feats = np.array(feats,dtype=object)

    min_count=200
    n_seeds = np.array([args.Seeds])#np.random.randint(0,10000,size=args.Seeds) # number of seeds for checking randomness
    K_range = np.arange(1,12,1) #range of delays
    n_clusters=np.arange(50,3000,200) #number of partitions to explore

    # bootstrap_range = (pca_fish.shape[0]//args.Div)*args.Div
    # divs = np.arange(0,bootstrap_range,args.Div)

    params = []
    for s in n_seeds:
        for k in K_range:
            for cs in n_clusters:
                params.append([s,k,cs])
    #print(params)

    h_K = ma.zeros((len(n_seeds),len(K_range),len(n_clusters)))
    calc_ent = functools.partial(calc_entropy_from_data, feats, args.Recs, lengths_all)

    results = []
    # for i in range(len(params)):
    #   results.append(calc_ent(params[i]))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(calc_ent, params)

    for result in results:

         s = np.where(n_seeds == result[0])[0][0]
         K = np.where(K_range == result[1])[0][0]
         cs = np.where(n_clusters == result[2])[0][0]
         H = result[3]
         h_K[s, K, cs] = np.asarray(H)[:]

    h_K = ma.array(h_K)
    h_K[h_K==0] = ma.masked

    print('done', flush=True)
        
    save_path = args.Out + args.DatasetName + '/Entropy/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('Saving results ...', flush=True)
    f = h5py.File(args.Out+ args.DatasetName + '/Entropy/Entropy_seeds_delays_clusters_seed{}.h5'.format(args.Seeds),'w')
    entropies_ = f.create_dataset('entropies',h_K.shape)
    entropies_[...] = h_K
    n_seeds_ = f.create_dataset('seeds',n_seeds.shape)
    n_seeds_[...] = n_seeds
    K_range_ = f.create_dataset('K_range',K_range.shape)
    K_range_[...] = K_range
    n_clusters_ = f.create_dataset('n_clusters',n_clusters.shape)
    n_clusters_[...] = n_clusters


    print('Ended run, total time taken was', time.time() - start_time, flush=True)


if __name__ == "__main__":
    main(sys.argv)

