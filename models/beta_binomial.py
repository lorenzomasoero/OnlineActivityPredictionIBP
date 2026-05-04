"""Beta-Binomial model (see Ionita-Laza 2009 PNAS)."""

import numpy as np
import scipy.stats as spst
from scipy.special import beta as spb
from scipy import optimize
from scipy.special import gammaln as gln
from scipy.special import betaln as bln
from scipy.special import binom as spbin
import time

from .common import log_poch


class BB():

    '''
        Class to fit and instantiate beta-Bernoulli model [see Ionita-Laza 2009 PNAS]
    '''

    def __init__(self):

        return

    def instantiate_BB(self, alpha, beta, T, N_ub=int(1e5), seed=0, store_matrix=False):

        self.alpha = alpha
        self.beta = beta
        self.N_ub = N_ub
        self.T = T
        if store_matrix == True:
            self.counts, self.fa, self.sfs, self.thetas, self.bin_mat = self.draw_BB(alpha, beta, seed, True)
        else:
            self.counts, self.fa, self.sfs, self.thetas = self.draw_BB(alpha, beta, seed)

    def draw_BB(self, alpha, beta, seed=0, store_matrix=False):
        '''
        Input:
            alpha <float >0> first parameter of beta prior
            beta <float >0> second parameter of beta prior
            seed <int> RNG
            store_matrix <bool>
        Output:
            counts <array> shape T+1: # distinct variants
            fa <array> shape N_ub: frequency for each variant
            sfs <array> site frequency spectrum
        '''

        np.random.seed(seed)
        freqs = np.random.beta(alpha, beta, size=self.N_ub)
        thetas = freqs[freqs > 0]
        bin_mat = np.random.binomial(1, np.repeat(thetas, self.T)).reshape(len(thetas), self.T).T
        counts = self.accumulation_curve(bin_mat)
        fa = bin_mat.sum(axis=0)
        sfs = np.bincount(fa)[1:]

        if store_matrix == True:
            return counts, fa.astype(int), sfs, thetas, bin_mat
        return counts, fa.astype(int), sfs, thetas

    def accumulation_curve(self, bin_mat):
        bin_mat_cumsum = bin_mat.cumsum(axis=0)
        return np.concatenate([[0], np.count_nonzero(bin_mat_cumsum, axis=1)]).astype(int)

    def expected_news(self, T, T_2, sfs_1, params):
        alpha, beta = params
        return sfs_1 * (beta+T-1)/(alpha*T) * (1 - np.exp(log_poch(beta+T, np.arange(1, T_2+1)) - log_poch(alpha+beta+T, np.arange(1, T_2+1))))

    def predict_counts_news(self, T, T2, sfs_1, bb_params, counts):

        return np.concatenate([counts, counts[-1]+self.expected_news(T, T2, sfs_1, bb_params)])

    '''

        Beta : fitting with likelihood

    '''

    def fit_likelihood(self, sfs, T, num_its=10, status=False):
        bnds = ((.00001, 1000), (.00001, 1000),)
        optimal_values_, optimal_params_ = np.zeros([num_its]), np.zeros([num_its, 2])

        cost = self.make_beta_likelihood(sfs, T)

        if status == True:
            for it in range(num_its):
                devol = optimize.differential_evolution(cost, bnds)
                optimal_params_[it] = devol.x
                optimal_values_[it] = devol.fun
        else:
            for it in range(num_its):
                devol = optimize.differential_evolution(cost, bnds)
                optimal_params_[it] = devol.x
                optimal_values_[it] = devol.fun
        opt_ind = np.argmin(optimal_values_)
        return optimal_params_[opt_ind]

    def make_beta_likelihood(self, sfs, T):

        sfs = np.asarray(sfs)
        N_tot = sfs.sum()

        def beta_likelihood(params):
            alpha, beta = params
            probabilities = spbin(T, np.arange(1, T+1)) * np.exp(bln(np.arange(1, T+1) + alpha, np.arange(1, T+1)[::-1]+beta)-bln(alpha, beta))
            log_p = np.log(probabilities/probabilities.sum())
            loglike = np.inner(log_p[:len(sfs)], sfs)
            return - loglike

        return beta_likelihood

    '''

        Beta : fitting with Geometric likelihood

    '''

    def fit_geometric_likelihood(self, sfs, T, num_its=10, status=False):
        bnds = ((.00001, 1000), (.00001, 1000),)
        optimal_values_, optimal_params_ = np.zeros([num_its]), np.zeros([num_its, 2])

        cost = self.make_beta_geometric_likelihood(sfs, T)

        if status == True:
            for it in range(num_its):
                devol = optimize.differential_evolution(cost, bnds)
                optimal_params_[it] = devol.x
                optimal_values_[it] = devol.fun
        else:
            for it in range(num_its):
                devol = optimize.differential_evolution(cost, bnds)
                optimal_params_[it] = devol.x
                optimal_values_[it] = devol.fun
        opt_ind = np.argmin(optimal_values_)
        return optimal_params_[opt_ind]

    def make_beta_geometric_likelihood(self, sfs, T):

        sfs = np.asarray(sfs)
        N_tot = sfs.sum()

        def beta_likelihood(params):
            alpha, beta = params
            probabilities = np.exp(log_poch(beta, np.arange(len(sfs))) - log_poch(alpha+beta, np.arange(len(sfs))))
            log_p = np.log(probabilities/probabilities.sum())
            loglike = np.inner(log_p[:len(sfs)], sfs)
            return - loglike

        return beta_likelihood

    '''

        Beta : fitting with Method of Moments

    '''

    def fit_method_of_moments(self, sfs, T):

        N = int(sfs.sum())
        mean = 1/(T*N) * np.dot(np.arange(1, len(sfs)+1), sfs)
        var = 1/(T*N**2) * np.dot(sfs, (np.arange(1, len(sfs)+1) - mean)**2)
        alpha = mean * (mean*(1-mean)/var - 1)
        beta = (1-mean)*(mean*(1-mean)/var - 1)

        return alpha, beta

    '''

        Beta : fitting with regression

    '''

    def make_cost_function(self, train_counts, from_, up_to, sfs_1, norm):

        def cost_function(params):
            predicted = train_counts[from_] + self.expected_news(from_, up_to - from_, sfs_1, params)
            delta = predicted - train_counts[from_:up_to]
            cost = np.linalg.norm(delta, ord=norm)
            return cost

        return cost_function

    def fit_regression(self, train_counts, sfs_1, num_its=10, norm=2, status=True):

        bnds = ((0, 10000), (0, 10000),)
        optimal_values_, optimal_params_ = np.zeros(num_its), np.zeros([num_its, 2])
        N = len(train_counts)
        from_, up_to = int(N * 1 / 5), N
        cost = self.make_cost_function(train_counts, from_, up_to, sfs_1, norm)
        if status == True:
            for it in range(num_its):
                devol = optimize.differential_evolution(cost, bnds)
                optimal_params_[it] = devol.x
                optimal_values_[it] = devol.fun
        else:
            for it in range(num_its):
                devol = optimize.differential_evolution(cost, bnds)
                optimal_params_[it] = devol.x
                optimal_values_[it] = devol.fun
        opt_ind = np.argmin(optimal_values_)
        return optimal_params_[opt_ind]
