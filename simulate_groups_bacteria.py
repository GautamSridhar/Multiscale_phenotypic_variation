#data format library
import h5py

#numpy
import numpy as np
import numpy.ma as ma
import sys
sys.path.append('./utils/') ## Update path here
import os
import copy
import partitioning_methods as cl
import operator_calculations as op_calc
import delay_embedding as embed
import stats
import time
import argparse
from joblib import Parallel, delayed


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

def simulate(P,state0,iters,lcs):
    states = np.zeros(iters,dtype=int)
    states[0]=state0
    state=state0
    for k in range(1,iters):
        new_state = np.random.choice(np.arange(P.shape[1]),p=list(np.hstack(P[state,:].toarray())))
        state=new_state
        states[k]=state
    return lcs[states]

def simulate_parallel(P,state0,len_sim,lcs):
    return simulate(P,state0,len_sim,lcs)

def rec_trajectory(dpsi,dist,psi0,X0):
    rec_psis = psi0+np.cumsum(dpsi)
    rec_vecs = ma.vstack([dist[1:]*np.cos(rec_psis[:-1]),dist[1:]*np.sin(rec_psis[:-1])]).T
    rec_X_traj = X0+np.cumsum(rec_vecs,axis=0)
    return np.vstack([X0,rec_X_traj])

def get_sims_group(labels_recs,lookup_table,recs_idx,dpsi,dist,psi_start,X_start,delay=1,n_sims=10, len_sim = 1000):
    labels_group = labels_recs.copy()
    labels_group[recs_idx] = ma.masked
    labels_group = ma.hstack(labels_group)

    segments = op_calc.segment_maskedArray(labels_group)
    dtrajs = np.asarray([labels_group[t0:tf] for t0,tf in segments],dtype=object)

    lcs,P = op_calc.transition_matrix(dtrajs,delay,return_connected=True)
    final_labels_state = op_calc.get_connected_labels(labels_group,lcs)
    unique_labels = np.unique(final_labels_state.compressed())
    states0 = np.random.choice(unique_labels,n_sims).astype(int)
    
    sims = Parallel(n_jobs=-1)(delayed(simulate_parallel)(P,state0,len_sim,lcs) for state0 in states0)
    
    sims_X_traj = []
    for sim in sims:
        dpsi_sim = np.array([dpsi[lookup_table[s][np.random.randint(0,len(lookup_table[s]))]] for s in sim])
        dist_sim = np.array([dist[lookup_table[s][np.random.randint(0,len(lookup_table[s]))]] for s in sim])
        sim_X_traj = rec_trajectory(dpsi_sim,dist_sim,psi_start,X_start)
        sims_X_traj.append(ma.array(sim_X_traj))
    return sims,sims_X_traj


def main(argv):
    t_0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-len_sim','--L',help="len_sim",default=1000,type=int)
    parser.add_argument('-idx','--Idx',help="idx",default=0,type=int)
    parser.add_argument('-dn','--DatasetName',help="Name of the dataset to analyze", default='filtered_phdata7_condition{}.h5',type=str)
    parser.add_argument('-out','--Out',help="path to save",default='/flash/StephensU/Gautam/HMD/Results/',type=str)
    args=parser.parse_args()
    idx = args.Idx

    print("Load data",flush=True)

    f = h5py.File('/flash/StephensU/Gautam/HMD/Results/'+args.DatasetName+ '/microstate_labels.h5','r')
    lengths_all = np.array(f['MetaData/lengths_data'], dtype=int)
    labels_recs = ma.array(f['microstate_labels'],dtype=int)
    f.close()
    to_mask = np.max(labels_recs.data)
    labels_recs[labels_recs == to_mask] = ma.masked
    labels_all= ma.concatenate(labels_recs,axis=0)

    
    path_to_filtered_data = '/flash/StephensU/Gautam/HMD/Results/bacteria_all2/'
    f= h5py.File(path_to_filtered_data+'bacteria_all3.h5','r')
    print(f.keys())
    bacteria_idx_all = np.array(f['MetaData/bacteria_idx_all'],dtype=int)
    X_head = ma.asarray(f['X_all'],dtype=float)

    X = ma.vstack(X_head.copy())

    vecX = ma.diff(X[:,:],axis=0)
    dist = ma.zeros(X.shape[0])
    dist[:-1] = ma.sqrt(vecX[:,0]**2+vecX[:,1]**2)
    dist[-1] = ma.masked 

    psi = ma.zeros(X.shape[0])
    psi[:-1] = ma.arctan2(vecX[:,1],vecX[:,0])
    psi[-1] = ma.masked

    psi_unwrap = unwrapma(psi)
    dpsi = ma.zeros(X.shape[0])
    dpsi[:-1] = psi_unwrap[1:]-psi_unwrap[:-1]
    dpsi[-1:] = ma.masked

    psi_clus = psi_unwrap.reshape(X_head[:].shape[0],X_head[:].shape[1])
    
    lookup_table = {}
    for state in np.unique(labels_all.compressed()):
        mask = labels_all==state
        lookup_table[state] = np.arange(len(labels_all))[mask]


    clus,idx = np.array(np.loadtxt(args.Out+args.DatasetName+'/sims/iteration_indices_groups.txt')[idx],dtype=int) ## Update Path here

    tmspace_clusters = np.load(args.Out+args.DatasetName + '/sims/spectral_split_g3.npy')
    print(np.unique(tmspace_clusters),flush=True)
    print(clus,idx,flush=True)
        
        
    print("Run simulations",flush=True)

    n_sims = 100
    len_sim = args.L

    print(clus,flush=True)
    rec_idx = np.arange(0,len(labels_recs))
    cluster_idx = np.asarray(np.where(tmspace_clusters == clus)[0], dtype=int)
    rec_idx = np.delete(rec_idx,cluster_idx)
    print(len(cluster_idx),flush=True)
    sims_gs,sims_gs_X_traj = get_sims_group(labels_recs,lookup_table,rec_idx,dpsi,dist,np.random.choice(psi_clus[:,0]), [0,0],
                                     n_sims=n_sims, len_sim=len_sim)

    sims_gs = np.array(sims_gs)
    sims_gs_X_traj = np.array(sims_gs_X_traj)
    
    print(sims_gs.shape,flush=True)
    print(sims_gs_X_traj.shape,flush=True)
    
    print("Save simulations",flush=True)

    f = h5py.File(args.Out +args.DatasetName+ '/sims/sims_lensim_{}_group_{}_{}.h5'.format(len_sim,clus,idx),'w')
    s_ = f.create_dataset('sims_gs',sims_gs.shape)
    s_[...] = sims_gs
    sX_ = f.create_dataset('sims_gs_X',sims_gs_X_traj.shape)
    sX_[...] = sims_gs_X_traj
    f.close()
    t_f = time.time()
    print('It took {:.2f} minutes.'.format((t_f-t_0)/60.))
    
if __name__ == "__main__":
    main(sys.argv)