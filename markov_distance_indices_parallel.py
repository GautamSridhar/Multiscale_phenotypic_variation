#data format library
import h5py
#numpy
import numpy as np
import numpy.ma as ma
import sys
sys.path.append('./utils/')
import os
import partitioning_methods as cl
import operator_calculations as op_calc
import stats
import time
from scipy.sparse import csr_matrix
import deeptime.markov.tools.estimation as dt_estimation
import argparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


def _transitions_from_labels(labels, lag=1):
    a = labels[:-lag]; b = labels[lag:]
    if isinstance(labels, np.ma.MaskedArray):
        m = ~(np.ma.getmaskarray(a) | np.ma.getmaskarray(b))
        a = np.asarray(a)[m]; b = np.asarray(b)[m]
    else:
        a = np.asarray(a); b = np.asarray(b)
    return a, b

def row_normalize_dense_inplace(C):
    """
    In-place row normalization with zero-row protection.
    Leaves all-zero rows unchanged. (Exact same policy as before.)
    """
    C = np.asarray(C, dtype=np.float64, order="C")  # keep/ensure float64
    row_sums = C.sum(axis=1, keepdims=True)
    np.divide(C, row_sums, out=C, where=row_sums != 0)
    return C

def MC_distance(P1, P2):
    return 0.5 * np.mean(np.sum(np.abs(P1 - P2), axis=1))

def getP(labels, nstates, lag=1):
    """Dense, fast P via bincount; identical result to your original."""
    r, c = _transitions_from_labels(labels, lag=lag)
    idx = np.ravel_multi_index((r, c), (nstates, nstates))
    counts = np.bincount(idx, minlength=nstates * nstates).astype(np.float64)
    counts = counts.reshape(nstates, nstates)
    return row_normalize_dense_inplace(counts)


def get_perms(n_all, n_boot=1000):
    """
    Vectorized: n_boot independent permutations of range(n_all).
    Each row is a permutation (without replacement).
    """
    rng = np.random.default_rng()
    keys = rng.random((n_boot, n_all), dtype=np.float64)  # i.i.d. keys
    perms = np.argsort(keys, axis=1)                      # each row is a permutation
    return perms.astype(np.int64)


def eps_helper_coarse(perms, k, n_all, idx_all, nstates, nstates_coarse, group_labels, n_boot=None):
    """
    Vectorized replacement for the loop version:
      - counts for both groups with batch bincount (no Python loops)
      - same result as your original implementation.
    Signature kept for drop-in compatibility. n_all/n_boot are unused.
    """
    B = perms.shape[0]
    M = nstates_coarse * nstates_coarse

    idx_i = group_labels[idx_all // nstates]
    idx_j = group_labels[idx_all %  nstates]
    coarse_flat_all = (idx_i * nstates_coarse + idx_j).astype(np.int64) 

    # Split each permutation into the two groups of positions
    g1 = coarse_flat_all[perms[:, :k]]           # shape (B, k)
    g2 = coarse_flat_all[perms[:, k:]]           # shape (B, n_all-k)

    # Batch offsets so we can count all bootstraps in one np.bincount call
    offsets = (np.arange(B, dtype=np.int64)[:, None]) * M  # (B, 1)
    idx1 = (g1 + offsets).ravel()
    idx2 = (g2 + offsets).ravel()

    # Counts per bootstrap -> reshape to (B, nstates_coarse, nstates_coarse)
    counts1 = np.bincount(idx1, minlength=B * M).reshape(B, nstates_coarse, nstates_coarse).astype(np.float64)
    counts2 = np.bincount(idx2, minlength=B * M).reshape(B, nstates_coarse, nstates_coarse).astype(np.float64)

    B, q, _ = counts1.shape
    counts1_2d = counts1.reshape(B * q, q)
    counts2_2d = counts2.reshape(B * q, q)
    
    row_normalize_dense_inplace(counts1_2d)  # this expects (rows, cols)
    row_normalize_dense_inplace(counts2_2d)
    
    Ps1 = counts1_2d.reshape(B, q, q)
    Ps2 = counts2_2d.reshape(B, q, q)

    # Distances per bootstrap (identical formula)
    dists = 0.5 * np.mean(np.sum(np.abs(Ps1 - Ps2), axis=2), axis=1)
    return dists


def get_dists_coarse(i, j, labels_sims, n_states, q_range, group_labels_q, n_boot=1000, lag=1):
    labels1 = labels_sims[int(i)]
    labels2 = labels_sims[int(j)]
    r1, c1 = _transitions_from_labels(labels1, lag)
    r2, c2 = _transitions_from_labels(labels2, lag)
    k = r1.size
    r_all = np.concatenate([r1, r2])
    c_all = np.concatenate([c1, c2])
    n_all = r_all.size

    # Fine flat transitions (only needed to compute coarse bins above)
    idx_all = np.ravel_multi_index((r_all, c_all), (n_states, n_states))

    # Vectorized permutations for all bootstraps (without replacement)
    perms = get_perms(n_all, n_boot=n_boot)

    Ds_q = np.empty((len(q_range),2), dtype=np.float64)
    for kq, q in enumerate(q_range):
        gl = np.asarray(group_labels_q[kq], dtype=np.int64)
        labels1_ = labels1.copy()
        labels1_ = ma.filled(labels1_,0)
        gl1 = ma.masked_invalid(gl)[labels1_]
        gl1[labels1.mask]=ma.masked
        labels2_ = labels2.copy()
        labels2_ = ma.filled(labels2_,0)
        gl2 = ma.masked_invalid(gl)[labels2_]
        gl2[labels2.mask]=ma.masked
        P1 = getP(gl1,q,lag)
        P2 = getP(gl2,q,lag)
        Ds_q[kq,0] = MC_distance(P1,P2)
        dists = eps_helper_coarse(perms, k, n_all, idx_all, n_states, q, gl)
        Ds_q[kq,1] = dists.mean()
    return Ds_q



def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='idx',default=0,type=int)
    parser.add_argument('-t','--Tau', help='Delay of the PF operator', type=int, default=1)
    parser.add_argument('-nboot','--NumBoot',help="Number of bootstrap estimates to perform",type=int,default=100)
    parser.add_argument('-dn','--DatasetName',help="Name of the dataset to analyze", default='filtered_phdata7_condition{}.h5',type=str)
    parser.add_argument('-out','--Out',help="path to save",default='/flash/StephensU/Gautam/Multiscale_phenotypic_variation/Results/',type=str)
    args=parser.parse_args()

    idx = int(args.Idx)
    indices = np.loadtxt(args.Out + '/indices/all_indices_{}.txt'.format(idx))
    pairs = np.array(indices,dtype=int)
    
    print(indices.shape,flush=True)

    print('Get coarse labels',flush=True)

    f = h5py.File(args.Out + args.DatasetName + '/microstate_labels.h5')
    # labels_recs = ma.array(f['labels_fish'],dtype=int)  ## Uncomment for zebrafish
    labels_recs = ma.array(f['microstate_labels'],dtype=int) ## Comment out for zebrafish
    lengths_all = ma.array(f['MetaData/lengths_data'],dtype=int)
    f.close()

    to_mask = np.max(labels_recs.data)
    labels_recs[labels_recs==to_mask] = ma.masked

    P_ensemble = np.load('/flash/StephensU/Gautam/HMD/Results/'+args.DatasetName+'/P_ensemble.npy')
    P_ensemble = csr_matrix(P_ensemble)

    lcs_ensemble = dt_estimation.largest_connected_set(P_ensemble)
    labels_all = ma.hstack(labels_recs)
    final_labels = op_calc.get_connected_labels(labels_all, lcs_ensemble)
    final_labels_recs = final_labels.reshape(labels_recs.shape)
    
    n_states_all = len(lcs_ensemble)
    print(n_states_all)
    q_range = np.unique(np.array(np.logspace(0,2,50),dtype=int))[1:]  
    # q_range = np.unique(np.array(np.logspace(0,2.3,75),dtype=int))[1:] ## Uncomment for zebrafish and bacteria results

    f= h5py.File('/flash/StephensU/Gautam/HMD/Results/'+args.DatasetName+'/cg_labels_tree_minrho.h5','r')
    labels_tree = np.array(f['labels_tree'],dtype=int)
    f.close()

    group_labels_q = []
    for q in q_range:
        group_labels = labels_tree[q-2]
        group_labels_q.append(group_labels)

    print('Estimate distances',flush=True)

    pairs = indices

    print(len(pairs),args.Tau,labels_recs.max(),len(group_labels_q[0]),flush=True)

    t_0 = time.time()
    Ds_q_indices = Parallel(n_jobs=-1)(
        delayed(get_dists_coarse)(i, j, final_labels_recs, n_states_all, q_range, group_labels_q, args.NumBoot, args.Tau)
        for i, j in pairs
    )
    t_f = time.time()

    print('Loop took {:.2f} mins'.format((t_f-t_0)/60.),flush=True)
    
    Ds_q_indices = np.array(Ds_q_indices)

    print(Ds_q_indices.shape)

    if not os.path.exists(args.Out+args.DatasetName+'/Eps_parallel/'):
        os.makedirs(args.Out+args.DatasetName+'/Eps_parallel/')

    print('Saving results',flush=True)
    f= h5py.File(args.Out+args.DatasetName+'/Eps_parallel/Ds_q_idx_{}.h5'.format(idx),'w')
    Dc_ = f.create_dataset('Ds_q',Ds_q_indices.shape)
    Dc_[...] = Ds_q_indices
    q_range_ = f.create_dataset('q_range',q_range.shape)
    q_range_[...] = q_range
    f.close()
    
    print('Saved results',flush=True)
    
if __name__ == "__main__":
    main(sys.argv)
