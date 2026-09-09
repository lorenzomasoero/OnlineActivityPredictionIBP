import numpy as np
from scipy.stats import nbinom as nbinom
from scipy.special import beta as betafn, gammaln as gln, betaln as bln, binom as spbin
import time
from tqdm.notebook import tqdm
from scipy import optimize
from scipy.stats.mstats import winsorize

from matrix_utils import *


class NegBintSBSP():
    
    '''
        In this class, we implement the NegativeBinomial stable-beta-scaled-process 
    '''
    
    def __init__(self):
        
        return

    def instantiate(self,
                    D, # total number of experimental days
                    parameters, # (beta, sigma, tilting, r)
                    seed, # random seed
                    store_matrix # True to return matrix
                   ):
        
        beta, sigma, tilting, r = parameters
        self.beta = beta
        self.sigma = sigma
        self.tilting = tilting
        self.r = r
        self.D = D
        self.seed = seed
        self.store_matrix = store_matrix
            

    def draw(self,
             seed = None
            ):
        
        '''
            Produces a draw from a negative-binomial stable-beta-scaled-process prior.
            
            Output:
                total_number_of_users <array> of size D+1; new users accumulation curve
                num_new_daily_users <array> of size D; for each day the number of new users triggering; 
                unit_level_total_retrigger <array> of size N (total number of users); each entry is total re trigger rate
                matrix <array> of size D * N; [d,n]-th entry is # re triggers of unit n on day d
        '''
        
        
        self.parameters = self.beta, self.sigma, self.tilting, self.r # unfold parameters
        if seed == None:
            np.random.seed(self.seed) # for replicability and pseudorandomness 
        else:
            np.random.seed(seed)
        num_new_daily_users = np.zeros(self.D, dtype = int) # initialize empty vector with daily counts of new users
        total_users_seen_so_far = 0
        for d in range(1, self.D+1): # for every day in the experiment
            
            num_new_daily_users[d-1] = int(nbinom.rvs(self.tilting + total_users_seen_so_far+1, 
                                                      1-self.get_proba_news(d-1, 1, self.parameters)))
            total_users_seen_so_far += num_new_daily_users[d-1]
            
        total_number_of_users = np.concatenate([[0], num_new_daily_users.cumsum().astype(int)])
        
        
        # now we populate the matrix of re-triggers
        # for each user, we provide a running count of the total number of times they re-triggered in the experiment
        
        unit_level_total_retrigger = np.zeros(total_number_of_users[-1]) # one value per user
        
        if self.store_matrix ==  True:
            matrix = np.zeros([self.D, total_number_of_users[-1]], dtype = int)
        
        for d in tqdm(range(1, self.D+1)):
            
            number_active_users_before_day_d = total_number_of_users[d-1]
            trigger_counts_old_users_previous_visits = unit_level_total_retrigger[:number_active_users_before_day_d]
            trigger_counts_old_users_at_d = self.sample_retrigger_counts_previously_seen(d-1,
                                                                                         1,
                                                                                         trigger_counts_old_users_previous_visits, 
                                                                                         self.parameters,
                                                                                         N_MC = 1
                                                                                  ).flatten()
            num_new_users_on_day_d = num_new_daily_users[d-1] # notice because of python indexing we have to look at d-1 
            trigger_counts_new_users_at_d = self.sample_retrigger_counts_new(d,
                                                                             num_new_users_on_day_d,
                                                                             self.parameters)
            
            if self.store_matrix: # populate the d-1 th row of the matrix with the actual values
                                
                matrix[d-1, :total_number_of_users[d-1]] = trigger_counts_old_users_at_d
                matrix[d-1, total_number_of_users[d-1]:total_number_of_users[d]] = trigger_counts_new_users_at_d

            unit_level_total_retrigger[:total_number_of_users[d-1]] += trigger_counts_old_users_at_d
            unit_level_total_retrigger[total_number_of_users[d-1]:total_number_of_users[d]] = trigger_counts_new_users_at_d
            
        if self.store_matrix: 

            return total_number_of_users, num_new_daily_users, unit_level_total_retrigger.astype(int), matrix
        
        return total_number_of_users, num_new_daily_users, unit_level_total_retrigger.astype(int)    
    
    ### sampling counts
    
    def sample_retrigger_counts_new(self,
                                    d,
                                    num_samples,
                                    parameters):
        beta, sigma, tilting, r = parameters

        max_val = 100
        unnormalized_pmf = spbin(np.arange(1, max_val) + r - 1, np.arange(1, max_val))[np.newaxis, :] * \
                           betafn(np.arange(1, max_val)-sigma, r * d + 1)  
        pmf = unnormalized_pmf/unnormalized_pmf.sum()

        return np.array([int(1+np.random.choice(np.arange(pmf.shape[1]), 
                                              size = 1, 
                                              replace = True, 
                                              p=p)) for p in pmf])     
    
    def sample_retrigger_counts_previously_seen(self,
                                                D_pilot, # int
                                                D_follow,
                                                counts_previous_retriggers, # array of ints
                                                parameters, 
                                                N_MC = 1234 
                                                ):
        '''
        Input:
            D_pilot <int> number of prior days of experimentation
            counts_previous_retriggers <array> number of retriggers among users who
                                               have already triggered in the first D_0 days
            parameters <tuple> beta, sigma, tilting, r
         Output:
             has shape <N_MC, D_follow, len(counts_previous_triggers)>           
        '''
        beta, sigma, tilting, r = parameters

        if len(counts_previous_retriggers) == 0:
            return np.array([])
        
        assert(counts_previous_retriggers>0).all(), 'Cannot sample re-triggers for users who yet have to trigger'
        
        
        jumps = np.random.beta(counts_previous_retriggers - sigma, r*D_pilot + 1, 
                               size = len(counts_previous_retriggers))
        return np.array([[nbinom.rvs(r * d, 1 - jumps) for d in range(1, D_follow+1)] for it in range(N_MC)])
    
    ### NEWS

    def get_proba_news(self,
                       D_pilot, # int
                       D_follow, # int
                       parameters # 
                      ):
        '''
            Computes the parameter that drives the number of new users triggering in the experiment
            p_{D_pilot}^{(D_follow)}
        '''
        beta, sigma, tilting, r = parameters
        return self.get_psi(D_pilot, D_follow, parameters) / \
                (beta+self.get_psi(0, D_pilot+D_follow, parameters))
            
    def get_proba_news_freq(self,
                            D_pilot, # int
                            D_follow, # int
                            parameters,
                            frequency # int
                           ):
        '''
            Computes the parameter that drives the number of new users triggering in the experiment with a given frequency
            p_{D_pilot}^{(D_follow, frequency)}
        '''
 
        beta, sigma, tilting, r = parameters

        return self.get_rho(D_pilot, D_follow, parameters, frequency) / \
                (beta+self.get_psi(0, D_pilot, parameters) + self.get_rho(D_pilot, D_follow, parameters, frequency))
    
    def get_psi(self, D_0, D_1, parameters):

        beta, sigma, tilting, r = parameters
        return sigma*(betafn(r*D_0+1, - sigma) - betafn(r*(D_0+D_1)+1, - sigma))
    
    def get_rho(self, D_0, D_1, parameters, frequency):
        
        beta, sigma, tilting, r = parameters
        return sigma * spbin(frequency + r*D_1 + 1, frequency) * betafn(r*(D_0+D_1) + 1, frequency - sigma) 
       
    ### STATISTICS 
       
    def mean_number_new_users(self,
                              D_pilot,
                              D_follow,
                              N_at_D_pilot,
                              parameters,
                              frequency = None):
        '''
        
        Computes the mean number U_{D_pilot}^{(D_follow, frequency)} of new users to be observed in D_follow 
        additional sampling days and retriggering exactly frequency many times,
        given data from the inital D_pilot days during which N_at_D_pilot users were already observed.
        NOTICE: If frequency = None, U_{D_pilot}^{(D_follow)} is computed.
         
        
        Input:
            D_pilot <int> number of prior days of experimentation
            D_follow <int> number of follow up days of experimentation
            N_at_D_pilot <int> number of unique active customers so far
            r <float>
            parameters <list> beta, sigma, tilting, r
            frequency if None, takes union <int>
        Output:
            has shape D_follow           
        '''
        beta, sigma, tilting, r = parameters
        
        if frequency == None:
            return np.array([
                                nbinom.mean(
                                            tilting + N_at_D_pilot + 1, 
                                            1-self.get_proba_news(D_pilot, d, parameters)
                                            )
                                for d in range(1, D_follow+1)
                           ])

        return np.array([
                            nbinom.mean(
                                        tilting + N_at_D_pilot + 1, 
                                        1-self.get_proba_news_freq(D_pilot, d, parameters, frequency)
                                        ) 
                            for d in range(1, D_follow+1)
                        ])
    
    def ci_number_new_users(self,
                              D_pilot,
                              D_follow,
                              N_at_D_pilot,
                              parameters,
                              width = 0.95,
                              frequency = None):
        '''
        
        Computes the 100* width % centrered confidence intervals for the number U_{D_pilot}^{(D_follow, frequency)} 
        of new users to be observed in D_follow 
        additional sampling days and retriggering exactly frequency many times,
        given data from the inital D_pilot days during which N_at_D_pilot users were already observed.
        NOTICE: If frequency = None, U_{D_pilot}^{(D_follow)} is computed.
                 
        Input:
            D_pilot <int> number of prior days of experimentation
            D_follow <int> number of follow up days of experimentation
            N_at_D_pilot <int> number of unique active customers so far
            r <float>
            parameters <list> beta, sigma, tilting, r
            frequency if None, takes union <int>
        Output:
            has shape (D_follow, 2)
        '''
        beta, sigma, tilting, r = parameters
        if frequency == None:
            return np.array([
                                nbinom.interval(
                                            width,
                                            tilting + N_at_D_pilot + 1, 
                                            1-self.get_proba_news(D_pilot, d, parameters)
                                            )
                                for d in range(1, D_follow+1)
                           ])

        return np.array([
                            nbinom.interval(
                                        width,
                                        tilting + N_at_D_pilot + 1, 
                                        1-self.get_proba_news_freq(D_pilot, d, parameters, frequency)
                                        ) 
                            for d in range(1, D_follow+1)
                        ])
    
    def total_retriggers_old_users(self,
                                   D_pilot,
                                   D_follow,
                                   counts_previous_retriggers,
                                   parameters,
                                   N_MC = 1234
                                   ):
        '''
        Input:
            D_pilot <int> number of prior days of experimentation
            D_follow <int> number of follow up days of experimentation
            counts_previous_retriggers <array> number of retriggers per user
            parameters <tuple> beta, sigma, tilting, r
            N_MC <int> number of re-draws
        Output:
            Winsorized Monte Carlo estimates of total future re triggers of previously seen users, 
            i.e. S_{D_0}^{(d)} | Z_{1:D_0} for d in {1,2,...,D_follow}
            has shape N_MC, D_follow
        '''
        return self.sample_retrigger_counts_previously_seen(
                                                            D_pilot, # int
                                                            D_follow,
                                                            counts_previous_retriggers, # array of ints
                                                            parameters, 
                                                            N_MC).sum(axis = -1) # this has shape N_MC, D_follow
    
    
    def total_retriggers_new_users(self,
                                   D_pilot,
                                   D_follow,
                                   N_at_D_pilot,
                                   parameters,
                                   N_MC = 1234,
                                   threshold = 10
                                   ):
        '''
        Input:
            D_pilot <int> number of prior days of experimentation
            D_follow <int> number of follow up days of experimentation
            counts_previous_retriggers <array> number of retriggers per user
            parameters <tuple> beta, sigma, tilting, r
            N_MC <int> number of re-draws
        Output:
            Winsorized Monte Carlo estimates of total future re triggers of previously seen users, 
            i.e. S_{D_0}^{(d)} | Z_{1:D_0} for d in {1,2,...,D_follow}
            has shape D_follow
        '''
        
        news_day_frequency = np.array([self.mean_number_new_users(D_pilot,
                                                                  D_follow,
                                                                  N_at_D_pilot,
                                                                  parameters,
                                                                  frequency=frequency) for frequency in range(1, threshold+1)
                                        ])
        # news_day_frequency has shape threshold, D_follow
        return np.sum(np.arange(1, threshold+1)[:, np.newaxis] * news_day_frequency, axis = 0)
        
        
    def predict_counts_news(self,
                            D_pilot, 
                            D_follow,
                            N_at_D_pilot, 
                            counts,
                            parameters):

        observed_counts = counts[:D_pilot+1]
        predicted_counts = self.mean_number_new_users(D_pilot, 
                                                      D_follow, 
                                                      N_at_D_pilot, 
                                                      parameters)
        return np.concatenate([observed_counts, observed_counts[-1] + predicted_counts])
    

    def make_log_like(self,
                      matrix):
    
        matrix = np.asarray(matrix)
        D, N = matrix.shape
        abundance = np.bincount(np.ndarray.flatten(matrix))[1:]
        retrigger_counts = matrix.sum(axis=0)
    
        def log_like(parameters):
            
            beta, sigma, tilting, r = parameters
            
            log_like_val = N * np.log(sigma) + (tilting+1) * np.log(beta) - (N+tilting+1) * np.log(beta + self.get_psi(0, D, parameters))
            log_like_val += gln(N+tilting+1) - gln(tilting)
            log_like_val += np.inner(abundance, log_binom(np.arange(1, len(abundance)+1)+r-1, 
                                                          np.arange(1, len(abundance)+1)
                                                         )
                                    ) 
            log_like_val += bln(D*r+1, retrigger_counts-sigma+1).sum()

            return - log_like_val
    
        return log_like
    
    def make_approx_log_like(self,
                             sfs,
                             D):
    
        N = sfs.sum()        
    
        def log_like(parameters):
            
            beta, sigma, tilting, r = parameters
            
            log_like_val = N * np.log(sigma) + (tilting+1) * np.log(beta) - (N+tilting+1) * np.log(beta + self.get_psi(0, D, parameters))
            log_like_val += gln(N+tilting+1) - gln(tilting)
            log_like_val += np.inner(sfs, log_binom(np.arange(1, len(sfs)+1)+r-1, 
                                                          np.arange(1, len(sfs)+1)
                                                         )
                                    ) 
            log_like_val += np.array([sfs_*bln(D*r+1, i-sigma+1) for i, sfs_ in enumerate(sfs)]).sum()

            return - log_like_val
    
        return log_like
    
    def make_approx_log_like_fixed_r(self,
                                     sfs,
                                     D):
    
        N = sfs.sum()        
    
        def log_like(parameters):
            
            r = 1
            beta, sigma, tilting = parameters
            paramters__ = beta, sigma, tilting, r
                        
            log_like_val = N * np.log(sigma) + (tilting+1) * np.log(beta) - (N+tilting+1) * np.log(beta + self.get_psi(0, D, paramters__))
            log_like_val += gln(N+tilting+1) - gln(tilting)
            log_like_val += np.inner(sfs, log_binom(np.arange(1, len(sfs)+1)+r-1, 
                                                          np.arange(1, len(sfs)+1)
                                                         )
                                    ) 
            log_like_val += np.array([sfs_*bln(D*r+1, i-sigma+1) for i, sfs_ in enumerate(sfs)]).sum()

            return - log_like_val
    
        return log_like
    
    def fit_log_like(self, 
                     matrix, 
                     num_its=10, 
                     bnds = ((.01,100), (.1,.999), (.1,100), (.1,100),), 
                     status = False):
        
        optimal_values_, optimal_params_ = np.zeros([num_its]), np.zeros([num_its,4])

        cost = self.make_log_like(matrix)
        
        if status == True:
            for it in tqdm(range(num_its)):
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

    def fit_approx_log_like(self, 
                            sfs,
                            D,
                            num_its=10, 
                            bnds = ((.01,100), (.1,.999), (.1,100), (.1,100),), 
                            status = False):
        
        optimal_values_, optimal_params_ = np.zeros([num_its]), np.zeros([num_its,4])

        cost = self.make_approx_log_like(sfs, D)
        
        if status == True:
            for it in tqdm(range(num_its)):
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
    
    def fit_approx_log_like_fixed_r(self, 
                                    sfs,
                                    D,
                                    num_its=10, 
                                    bnds = ((.01,100), (.1,.999), (.1,100),), 
                                    status = False):
        
        optimal_values_, optimal_params_ = np.zeros([num_its]), np.zeros([num_its,3])

        cost = self.make_approx_log_like_fixed_r(sfs, D)
        
        if status == True:
            for it in tqdm(range(num_its)):
                devol = optimize.differential_evolution(cost, bnds)
                optimal_params_[it] = devol.x
                optimal_values_[it] = devol.fun
        else:
            for it in range(num_its):
                devol = optimize.differential_evolution(cost, bnds)
                optimal_params_[it] = devol.x
                optimal_values_[it] = devol.fun
        opt_ind = np.argmin(optimal_values_)
        
        return np.concatenate([optimal_params_[opt_ind], [1]])
    
    
    def make_cost_function_regression(self,   
                                      D0, 
                                      N0,
                                      observed_counts,
                                      norm = 2,
                                      frequency=None):
        '''
        Input : 
            D_0 < int > minitrain pilot day
            observed_counts < array of ints, len D_1 > observed values that the regression is going to target 
            N_at_D_0 < int > # observed distinct users up until D_0; this is 0 if D_0 = 0
            norm = int -- norm to be used
            frequency
        Output :
            cost_function <function>; this is Eqn (***); 
                Input : params beta, c, sigma
                Output : scalar loss
        '''
        
        steps_ahead = len(observed_counts)
        #print(frequency)

        def cost_function(parameters):

            '''
                Takes as input parameters and returns discrepancy of true counts and predicted counts;
            '''
            if frequency == None:
                
                predicted = N0 + self.mean_number_new_users(D0, 
                                                            steps_ahead, 
                                                            N0, 
                                                            parameters)

                
            else:

                # notice that if we are fitting the frequencies, we need the number of total distincts and the number of freqs
                predicted = self.mean_number_new_users(D0, 
                                                       steps_ahead, 
                                                       N0, 
                                                       parameters, 
                                                       frequency=frequency)
            
                
            delta = predicted - observed_counts
            cost = np.linalg.norm(delta, ord = norm)
            
            return cost
        
        return cost_function

    def fit_regression(self,
                       D0,
                       N0,
                       observed_counts,
                       norm = 2, 
                       frequency = None,
                       num_its = 10,
                       status = False):
        '''
        Input :
            train_counts < array of ints ; len N > 
            num_its < int > number of times to optimization is performed
            norm < int > loss chosen
            status <bool> print status
            frequency <int> determines which cost function to use
        Output :
            optimal_params <array> (boot_its * num_its * 3)
            optimal_values <array> (boot_its * num_its)
        '''

        bnds =  bnds = ((.01,100), (.01,.999), (.01,100), (1,100),)
        optimal_values_, optimal_params_ = np.zeros(num_its), np.zeros([num_its, 4])
        cost = self.make_cost_function_regression(D0, 
                                                  N0,
                                                  observed_counts,
                                                  norm = norm,
                                                  frequency = frequency)
        if status == True:
            for it in tqdm(range(num_its)):
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