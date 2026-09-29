#data format library
import h5py
#numpy
import numpy as np
import numpy.ma as ma
import sys
import argparse
sys.path.append('./utils/')
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import os
import partitioning_methods as cl
import coarse_graining as cgm
import operator_calculations as op_calc
import stats
import time


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne','--NumEigfs', help='Number of eigenvectors above the noise floor to use', type=int, default=1)
    parser.add_argument('-cg','--Scale',help="Max coarse graining scale to run to",type=int,default=100)
    parser.add_argument('-s','--Seeds',help="Number of seeds to run",type=int,default=200)
    parser.add_argument('-t','--Tau', help='Delay of the PF operator', type=int, default=1)
    parser.add_argument('-dt','--DelT',help='Sampling rate of the dynamics i.e frame rate of recording',type=float,default=1)
    parser.add_argument('-dn','--DatasetName',help="Name of the dataset to analyze", default='filtered_phdata7_condition{}.h5',type=str)
    parser.add_argument('-out','--Out',help="path save",default='/flash/StephensU/Gautam/Multiscale_phenotypic_variation/Results/',type=str)
    args=parser.parse_args()
    

    f = h5py.File('/flash/StephensU/Gautam/HMD/Results/' + args.DatasetName + '/microstate_labels.h5')
    # labels_recs = ma.array(f['labels_fish'],dtype=int) ## Uncomment for zebrafish
    labels_recs = ma.array(f['microstate_labels'],dtype=int)
    lengths_all = ma.array(f['MetaData/lengths_data'],dtype=int)
    f.close()

    to_mask = np.max(labels_recs.data)
    labels_recs[labels_recs==to_mask] = ma.masked

    P_ensemble = np.load('/flash/StephensU/Gautam/HMD/Results/'+args.DatasetName+'/P_ensemble.npy')
    P_ensemble = csr_matrix(P_ensemble)

    inv_measure = op_calc.stationary_distribution(P_ensemble)
    R = op_calc.get_reversible_transition_matrix(P_ensemble)
    eigvals,eigvecs = op_calc.sorted_spectrum(R,k=args.NumEigfs+1)
    sorted_indices = np.argsort(eigvals.real)[::-1]
    eigvals = eigvals[sorted_indices][1:].real
    eigvals[np.abs(eigvals-1)<1e-12] = np.nan
    eigvals[eigvals<1e-12] = np.nan

    eigfunctions = eigvecs[:,sorted_indices].real/np.linalg.norm(eigvecs[:,sorted_indices].real,axis=0)
    X = eigfunctions[:,1:args.NumEigfs+1]*eigvals[:args.NumEigfs]

    print('Coarse grain labels',flush=True)
    labels_tree = cgm.kmeans_split(X,P_ensemble,inv_measure,args.Scale,args.Seeds)

        
    print('Saving results',flush=True)

    f= h5py.File(args.Out + args.DatasetName +'/cg_labels_tree_minrho.h5','w')
    gl_ = f.create_dataset('labels_tree',labels_tree.shape)
    gl_[...] = labels_tree
    f.close()
    
    print('Saved results',flush=True)
    
if __name__ == "__main__":
    main(sys.argv)
