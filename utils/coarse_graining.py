#data format library
import h5py
#numpy
import numpy as np
import numpy.ma as ma
from scipy.sparse import issparse
import os
import partitioning_methods as cl
import operator_calculations as op_calc
import stats
import time
from scipy.sparse import csr_matrix
import deeptime.markov.tools.estimation as msm_estimation
import deeptime.markov.tools.analysis as msm_analysis
import argparse
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

from scipy.signal import find_peaks
import skfuzzy as fuzz


from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.cluster import KMeans


def kmeans_split(X,P_ensemble,inv_measure,scale,seed_length):
    """
    Perform coarse-graining for transition matrix using kmeans clustering over multiple seeds.

    X: Eigenvectors of the transfer operator uptil particular numer
    P_ensemble: Ensemble transition matrix
    inv_measure: Invariance measure of the transition matrix
    scale: coarse-graining scale to coarse-grain up to
    seed_length: Number of seeds to run over
    """
    q_range = np.arange(2,scale+1)

    labels_tree=np.zeros((len(q_range),len(X)),dtype=int)

    np.random.seed(42)
    seeds_torun = np.random.randint(0,10000,seed_length)


    for i,q in enumerate(q_range):
        print(q,flush=True)
        min_rhos = []
        st = time.time()
        for s in seeds_torun:
            kmeans = KMeans(n_clusters=q, random_state=s,n_init=1).fit(X,sample_weight=inv_measure)
            ha = kmeans.labels_

            rho_sets = [(inv_measure[ha==idx]@(P_ensemble[ha==idx,:][:,ha==idx])).sum()/inv_measure[ha==idx].sum()
                                  for idx in np.unique(ha)]
            min_rhos.append(np.min(rho_sets))
        s = seeds_torun[np.where(min_rhos == np.max(min_rhos))[0]]
        kmeans = KMeans(n_clusters=q, random_state=s[0] ,n_init=1).fit(X,sample_weight=inv_measure)
        print('Time taken',time.time() - st,flush=True)
        labels_tree[i,:] = kmeans.labels_

    return labels_tree