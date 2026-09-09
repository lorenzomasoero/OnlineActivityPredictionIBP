"""Hierarchical beta-geometric (HBG) paper benchmark."""

from pathlib import Path

import numpy as np


class HierarchicalBetaGeometric:
    """Fit the HBG model and predict the cumulative number of users."""

    def predict_cumulative_users(
        self,
        first_trigger_counts: np.ndarray,
        followup_days: int,
        *,
        unseen_population_multiplier: float = 10.0,
        seed: int | None = None,
    ) -> np.ndarray:
        """Run the Stan model used for the HBG comparison in the paper."""

        try:
            import arviz as az
            from cmdstanpy import CmdStanModel
        except ImportError as error:
            raise ImportError(
                "HBG requires the optional dependencies cmdstanpy and arviz"
            ) from error

        counts = np.asarray(first_trigger_counts, dtype=int)
        if counts.ndim != 1 or np.any(counts < 0):
            raise ValueError("first_trigger_counts must be a nonnegative vector")
        if followup_days < 1 or unseen_population_multiplier <= 0:
            raise ValueError("followup_days and unseen_population_multiplier must be positive")

        observed_users = int(counts.sum())
        data = {
            "Ndays": len(counts),
            "Nfollow": followup_days,
            "users_per_day": counts,
            "n_notrig": int(unseen_population_multiplier * observed_users),
        }
        stan_file = Path(__file__).with_name("stan") / "hier_beta_geom.stan"
        fit = CmdStanModel(stan_file=str(stan_file)).sample(
            data=data,
            chains=1,
            parallel_chains=1,
            iter_warmup=1_000,
            iter_sampling=2_000,
            show_progress=False,
            seed=seed,
        )
        posterior = az.from_cmdstanpy(fit)
        future_users = np.mean(
            az.extract(posterior, var_names="n_new_users").to_numpy(), axis=1
        )
        observed_curve = np.concatenate(([0], counts.cumsum()))
        return np.concatenate((observed_curve, observed_users + future_users))
