import os
import numpy as np
# from tqdm import *
import time
from scipy.special import gamma as spg
from scipy.special import beta as spb
from scipy.special import binom as spbin
from scipy.special import betaln as bln
from scipy.special import gammaln as gln

import scipy.stats as spst
from scipy import optimize

def make_color_dict(color):
    
    dict_ = {'patch_artist': True,
             'boxprops': dict(color=color, facecolor='w'),
             'capprops': dict(color=color),
             'flierprops': dict(color=color, markeredgecolor=color),
             'medianprops': dict(color='k'),
             'whiskerprops': dict(color=color)}
    
    return dict_

def log_poch(x,n):
    '''
    Input :
        x : <float>
        n : <int>
    Output :
        log_poch<float> log of Pochammer symbol of x of order n, i.e. ln(Gamma(x+n)/Gamma(x)) 
    Further details:
        http://mathworld.wolfram.com/PochhammerSymbol.html
    '''
    return gln(x+n) - gln(x)

def log_beta(alpha, beta):
    '''
        Returns the logarithm of the beta function with parameters alpha, beta
    '''

    return gln(alpha)+gln(beta)-gln(alpha+beta)

def create_folder(path):
    '''
    Input :
        path : <str>
    Output :
        if there is not folder in path, such folder is created
    '''
    
    if not os.path.exists(path):
        os.makedirs(path)
        
def generate_bin_matrix_from_freqs(thetas, T, seed = 0):
    '''
    Input:
        thetas <array> shape K; contains frequencies of variants in [0,1]
        T <int> number of experiment days
        seed <int> seed for RNG
    Output:
        X <array> shape T, N; binary, with X[t,n] = 1(n-th customer logs in the experiment on day t), X[t,n] ~ Bern(thetas[n])
    '''
    np.random.seed(seed)
    X = np.random.binomial(1, np.repeat(thetas, T)).reshape(len(thetas), T).T
    
    return X

def generate_cts_from_bin_mat(X):
    '''
    Input:
        X <array> shape T,N, binary valued
    Output:
        cts <array> shape T+1; t-th entry is # distinct customers observed in first t-1 days; first entry is 0
    '''
    return np.concatenate([[0], np.count_nonzero(X.cumsum(axis=0), axis = 1)])

def count_new_freq(binary_matrix, T, T2, r):
    
    '''
       Input:
        binary_matrix <array> shape N,K, binary valued
        T <int> number of days in training set
        T2 <int> number of days for extrapolation 
        r <int> frequency
    Output:
        cts <int> number of customers not observed in first T days and appearing with frequency r in T2 additional days
    '''    
    assert T+T2<=binary_matrix.shape[0], 'T+T2 <= # rows'
    # first T rows are training
    
    binary_cumsum = binary_matrix.cumsum(axis = 0)
    yet_to_be_seen =(binary_cumsum[T] == 0)
    seen_with_frequency_at_T2 = np.array([(binary_cumsum[T+t] == r) for t in range(T2)])

    return np.sum(seen_with_frequency_at_T2*yet_to_be_seen[np.newaxis,:], axis = 1)