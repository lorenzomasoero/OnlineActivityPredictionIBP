module BnpActivity

include("empirical_bayes.jl")
include("predictive.jl")

export fit_be_ssp,
       fit_tg_ssp,
       new_user_credible_band,
       new_user_distributions,
       sample_be_ssp,
       sample_dm_posterior,
       sample_tg_ssp_first_trigger_times

end
