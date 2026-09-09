import numpy as np
from scipy.special import gammaln as gln

def log_binom(a, b):
    '''
        computes log(a choose b)
    '''
    return gln(a+1) - gln(b+1) - gln(a-b+1)

def count_news_from_matrix(matrix, start = 0, end = None, frequency = None):
    '''
        Input:
            matrix <array> shape T, N
            start <int> has to not exceed T
            end <int> has to be larger than T
            frequency <int>
        Output
            count_freq <array> shape end-start if end provided, else has shape T-start

        Given as input an integer valued matrix, for every row between {start, ..., end} 
        it returns the total number of columns in the matrix
        which count 0 up to start and have cumulative count equal to a 
        given frequency from start to the correpsonding row 
    '''
    
    idxs_news = np.where(matrix[:start].sum(axis = 0) == 0)[0]
    submatrix = matrix[start:end, idxs_news]
    submatrix_cumsum = submatrix.cumsum(axis = 0)
    if frequency == None:
        return (submatrix_cumsum > 0).sum(axis = 1)
    
    return (submatrix_cumsum == frequency).sum(axis = 1)
    
def count_total_abundance_previously_seen(matrix, pilot, follow):
    '''
        Input:
            matrix <array> shape T, N
            pilot <int> has to not exceed T
            follow <int> has to be larger than T-pilot
        Output
            sum_array <array> shape follow

        Given as input an integer valued matrix, for every row between {start, ..., end} 
        it returns the total sum for the columns
        which have at least one non-zero 0 in the first pilot rows
    '''    
    submatrix = matrix[:pilot].reshape(pilot, matrix.shape[1]) # matrix up to pilot
    active = np.sum(submatrix.sum(axis = 0)>0)
    if active == 0:
        return 0
    return matrix[pilot:pilot+follow, :active].sum()

def count_total_abundance_yet_to_be_seen(matrix, pilot, follow):
    '''
        Input:
            matrix <array> shape T, N
            pilot <int> has to not exceed T
            follow <int> has to be larger than T-pilot
        Output
            sum_array <array> shape follow
    
        Given as input an integer valued matrix, for every row between {start, ..., end} 
        it returns the total sum for the columns
        which have all 0 values in the first pilot rows
    '''   
    submatrix = matrix[:pilot].reshape(pilot, matrix.shape[1])
    active = np.sum(submatrix.sum(axis = 0)>0)
    if active == 0:
        return 0
    return matrix[pilot:pilot+follow, active:].sum()
