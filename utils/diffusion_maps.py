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

from scipy.signal import find_peaks
import skfuzzy as fuzz
from joblib import Parallel, delayed
from numba import njit


def simulate(P,state0,iters):
    states = np.zeros(iters,dtype=int)
    states[0]=state0
    state=state0
    for k in range(1,iters):
        new_state = np.random.choice(np.arange(P.shape[1]),p=P[state,:])
        state=new_state
        states[k]=state
    return states


def simulate_parallel(P,state0,len_sim):
    return simulate(P,state0,len_sim)


def get_sims_ensemble(P, delay=1, n_sims=10, len_sim=100000):
    """
    Generate simulations from the ensemble operator
    """
    states0 = np.random.randint(0,len(P),n_sims)
    sims = Parallel(n_jobs=-1)(delayed(simulate_parallel)(P,state0,len_sim) for state0 in states0)

    return sims

def shuffle_kern(M,delay,n_times=100):

    sims= get_sims_ensemble(M,delay=delay,n_sims=n_times)
    max_eigs = []
    for i in range(n_times):
        s = sims[i]
        np.random.shuffle(s)
        P = op_calc.transition_matrix([s],delay=delay)
        eig,_ = spla.eigs(P,k=3)
        sorted_indices = np.argsort(eig.real)[::-1]
        eig = eig[sorted_indices].real
        max_eigs.append(eig[1])
    return max_eigs


def PCA_entropy(D,eps):
    print(eps)
    M = Diffusion_Tmat(D,eps,1)
    
    diff_eig,_ = np.linalg.eig(M)
    sorted_indices = np.argsort(diff_eig.real)[::-1]
    diff_eig = diff_eig[sorted_indices].real
    diff_eig[diff_eig<0] = 0.
    # diff_eig = diff_eig[1:]
    norm_diff_eig = diff_eig/diff_eig.sum()
    return (-1*np.sum(norm_diff_eig*np.log(norm_diff_eig+1e-6))/np.log(len(diff_eig)))


def PCA_entropy_scaling(D,epsilons=None):
    if epsilons is None:
        epsilons = 2**np.linspace(-40., 41., 100)
    epsilons = np.sort(epsilons).astype('float')
    S = [PCA_entropy(D,eps) for eps in epsilons]
    # plt.plot(S,eps)
    return S,epsilons



def self_tuning_diffusion_tmat(D,sorted_d,k=1,alpha=1):
    sigma = np.zeros(len(D))
    for i in range(len(D)):
        nonzero = sorted_d[i, sorted_d[i] > 0]
        if len(nonzero) >= k:
            sigma[i] = nonzero[k - 1]
        else:
            sigma[i] = nonzero[-1] if len(nonzero) > 0 else 1.0  # fallback

    eps_ = np.outer(sigma,sigma)
    
    kern = np.exp(-(D**2)/(eps_))

    qa = (kern.sum(axis=0))**(-alpha)
    Da = np.diag(qa)
    L = np.dot(Da,np.dot(kern, Da))
    M = np.dot(np.diag(np.sum(L,axis=1)**-1),L)

    return M



def Diffusion_Tmat(D,eps,alpha=1):
    """
    Returns the transition matrix of a diffusion process calculated from a distance matrix
    """
    kern = np.exp(-(D**2)/(2*eps))

    qa = (kern.sum(axis=0))**(-alpha)
    Da = np.diag(qa)
    L = np.dot(Da,np.dot(kern, Da))
    M = np.dot(np.diag(np.sum(L,axis=1)**-1),L)

    return M


@njit(parallel=True, fastmath=True)
def loo_preds_numba(A, y, D2, eps_reg):
    n, p1 = A.shape
    y_pred = np.zeros(n)
    for i in prange(n):
        w = np.exp(-D2[i] / (eps_reg**2))
        w[i] = 0.0  # leave-one-out
        Aw = A * w.reshape(-1, 1)
        M = np.dot(A.T,Aw)
        b = np.dot(A.T, (w * y))
        lam = 1e-6 * np.trace(M) / M.shape[0]
        for j in range(M.shape[0]):
            M[j, j] += lam
        coeffs = np.linalg.lstsq(M, b, rcond=1e-12)[0]
        row = np.ascontiguousarray(A[i])
        coeffs = np.ascontiguousarray(coeffs)
        y_pred[i] = np.dot(row,coeffs)
    return y_pred


def local_linear_regression_reg(X, y, reg_range):
    n, p = X.shape
    A = np.column_stack([np.ones(n), X])  # (n, p+1)
    D2 = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
    r_eps = np.zeros(len(reg_range))
    for keps, eps_reg in enumerate(reg_range):
        y_pred = loo_preds_numba(A, y, D2, eps_reg)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y)**2)
        r_eps[keps] = np.sqrt(ss_res/ss_tot)
    return r_eps
