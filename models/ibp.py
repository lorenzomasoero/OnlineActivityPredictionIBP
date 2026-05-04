"""IBP (Indian Buffet Process) model."""

import numpy as np
import scipy.stats as spst
from scipy.special import beta as spb
from scipy import optimize
from scipy.special import gammaln as gln
from scipy.special import betaln as bln
from scipy.special import binom as spbinom
import time

from .common import log_poch

'''

    IBP DRAW FROM THE PRIOR

'''


class IBP():

    def __init__(self):

        return

    def instantiate_IBP(self, alpha, c, sigma, N, seed=0, store_matrix=False):

        self.alpha = alpha
        self.c = c
        self.sigma = sigma
        self.N = N
        if store_matrix == True:
            self.counts, self.unseen, self.fa, self.sfs, self.bin_mat = self.draw_IBP(alpha, c, sigma, seed, True)
        else:
            self.counts, self.unseen, self.fa, self.sfs = self.draw_IBP(alpha, c, sigma, seed)


    def draw_IBP(self, alpha, c, sigma, seed=0, store_matrix=False):

        np.random.seed(seed)
        unseen = np.zeros(1+self.N, dtype=int)  # number of news per step
        for n in range(1, self.N+1):
            p = self.parameter_new(alpha, c, sigma, n)
            newz = np.random.poisson(p)
            unseen[n] = newz
        counts = unseen.cumsum().astype(int)  # cumulative sum of news
        fa = np.zeros(counts[-1], dtype=int)
        low = counts[0]
        fa[:low] = 1  # these are the features sampled at first round

        if store_matrix == True:
            bin_mat = np.zeros([self.N, counts[-1]], dtype=int)
            bin_mat[0, :low] = 1

        for n in range(1, self.N):
            hi = counts[n+1]
            biases_0 = fa[:low]-sigma
            biases_1 = n-fa[:low]+c+sigma
            J_old = np.random.beta(biases_0, biases_1)
            xnk_old = np.random.binomial(n=1, p=J_old)
            fa[:low] += xnk_old
            fa[low:hi] = 1

            if store_matrix == True:
                bin_mat[n, :low] = xnk_old
                bin_mat[n, low:hi] = 1

            low = hi

        sfs = np.bincount(fa.astype(int))[1:]

        if store_matrix == True:
            return counts, unseen, fa.astype(int), sfs, bin_mat

        return counts, unseen, fa.astype(int), sfs

    def parameter_new(self, alpha, c, sigma, n):
        '''
        Input:
            alpha, c, sigma: parameters of 3 BP
            n <int>
        Output:
            expected number of new variants that n-th sample is displaying
        '''
        assert alpha > 0 and c > -sigma and sigma >= 0 and sigma < 1
        return alpha * np.exp(log_poch(c+sigma, n-1) - log_poch(c+1, n-1))

    def mean(self, N, M, params):
        alpha, c, sigma = params
        return alpha * np.exp(log_poch(c+sigma, np.arange(N, N+M)) - log_poch(c+1, np.arange(N, N+M))).cumsum()

    def mean_freq(self, N, M, r, params):

        alpha, c, sigma = params
        return alpha*spbinom(M, r) * np.exp(log_poch(1-sigma, r-1) + log_poch(c+sigma, N+np.arange(1, M+1)-r) - log_poch(c+1, N+np.arange(1, M+1)-1))

    def credible_interval(self, N, M, parameters, width):
        '''
        Input :
            N <int> number of samples you've seen so far
            M <int> number of additional samples;
            parameters <array> alpha, c, sigma
            width : % coverage of credible interval
        Output :
            lo, hi : lower and upper bounds of credible interval
        '''

        news = self.mean(N=N, M=M, params=parameters)
        lo, hi = spst.poisson.interval(alpha=width, mu=news)

        return lo, hi

    def credible_interval_freq(self, N, M, r, parameters, width):

        news = self.mean_freq(N=N, M=M, r=r, params=parameters)
        lo, hi = spst.poisson.interval(alpha=width, mu=news)

        return lo, hi

    def predict_counts_news(self, N, M, params, counts):
        K = counts[-1]
        return np.concatenate([counts, K+self.mean(N, M, params)])

    '''

        IBP OPTIMIZE

    '''

    def fit_EFPF(self, sfs, N, num_its=10, status=False):

        bnds = ((0, 100000), (0, 1000), (.00001, .9999),)
        optimal_values_, optimal_params_ = np.zeros([num_its]), np.zeros([num_its, 3])

        cost = self.make_EFPF(sfs, N)
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

    def make_EFPF(self, sfs, N):
        '''
        Input :
            sfs < array of ints, len K > true distinct counts
            N < int > total number of observation
        Output :
            cost_function <function>
        '''

        K = sfs.sum()

        def EFPF(params):
            alpha, c, sigma = params
            cost = K * (np.log(alpha) - log_poch(c+1, N-1)) - alpha * np.exp(log_poch(c+sigma, np.arange(N)) - log_poch(c+1, np.arange(N))).sum() + np.inner(sfs, log_poch(1-sigma, np.arange(len(sfs)))) + np.inner(sfs, log_poch(c+sigma, np.arange(N - len(sfs), N)[::-1]))
            return - cost

        return EFPF

    '''
        IBP: FITTING MEAN WITH REGRESSION
    '''

    def make_cost_function(self, train_counts, from_, up_to, norm):

        def cost_function(params):
            predicted = train_counts[from_] + self.mean(from_, up_to - from_, params)
            delta = predicted - train_counts[from_:up_to]
            cost = np.linalg.norm(delta, ord=norm)
            return cost

        return cost_function

    def regression(self, train_counts, num_its, norm, status):

        bnds = ((10e-9, 10e4), (0, 10), (.00001, .99999),)
        optimal_values_, optimal_params_ = np.zeros(num_its), np.zeros([num_its, 3])
        N = len(train_counts)
        from_, up_to = 0, N
        cost = self.make_cost_function(train_counts, from_, up_to, norm)
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
