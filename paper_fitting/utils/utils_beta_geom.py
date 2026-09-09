import numpy as np
from cmdstanpy import CmdStanModel
import arviz as az
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

import os
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
STAN_FILE = os.path.join(THIS_DIR, "hier_beta_geom_model.stan")

def predict_beta_geom(first_counts_short, D, Dfollow, lam=10):
    assert(len(first_counts_short) == D)
    n_seen = sum(first_counts_short)
    hgb_data = {
        "Ndays": len(first_counts_short),
        "Nfollow": Dfollow,
        "users_per_day": first_counts_short,
        "n_notrig": lam * n_seen
    }

    hbg = CmdStanModel(stan_file=STAN_FILE)
    model_fit = hbg.sample(data=hgb_data, chains=1, parallel_chains=1, 
                        iter_warmup=1000, iter_sampling=2000, show_progress=False)
    chains = az.from_cmdstanpy(model_fit)
    n_new = np.mean(az.extract(chains, var_names="n_new_users").to_numpy(), axis=1)
    
    counts_short_ = np.concatenate([[0], first_counts_short.cumsum()])
    
    return np.concatenate([counts_short_, n_seen + n_new])