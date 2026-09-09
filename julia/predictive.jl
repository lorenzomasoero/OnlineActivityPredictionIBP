using Distributions
using SpecialFunctions: beta


"""Sample first trigger times for users unseen before ``pilot_days``."""
function sample_tg_ssp_first_trigger_times(
    observed_first_trigger_days,
    alpha,
    c,
    beta_parameter,
    pilot_days::Integer,
    followup_days::Integer,
)
    observed_users = length(observed_first_trigger_days)
    support = (pilot_days + 1):(pilot_days + followup_days)
    probabilities = beta.(1 - alpha, support)
    normalizing_constant = sum(probabilities)
    probabilities ./= normalizing_constant

    followup_psi = alpha * sum(beta.(1 - alpha, support))
    pilot_psi = alpha * sum(beta.(1 - alpha, 1:pilot_days))
    probability = 1 - followup_psi / (
        beta_parameter + followup_psi + pilot_psi
    )

    number_new_users = rand(
        NegativeBinomial(observed_users + c + 1, probability),
    )
    return rand(Categorical(probabilities), number_new_users) .+ pilot_days
end


"""Sample binary activity from the paper's Be-SSP generative scheme."""
function sample_be_ssp(alpha, c, beta_parameter, number_days)
    cumulative_users = 0
    activity_matrix = zeros(0, 0)

    for day in 1:number_days
        beta_evaluations = beta.(1 - alpha, 1:day)
        numerator = alpha * beta_evaluations[end]
        denominator = beta_parameter + alpha * sum(beta_evaluations)
        probability = 1 - numerator / denominator
        number_new_users = rand(
            NegativeBinomial(cumulative_users + c + 1, probability),
        )

        if day == 1
            while number_new_users == 0
                number_new_users = rand(
                    NegativeBinomial(
                        cumulative_users + c + 1,
                        probability,
                    ),
                )
            end
        end

        cumulative_users += number_new_users
        if day == 1
            activity_matrix = ones(1, number_new_users)
            continue
        end

        active_day_counts = sum(activity_matrix, dims=1)
        new_row = rand.(
            Bernoulli.((active_day_counts .- alpha) ./ (day - alpha)),
        )'
        if number_new_users > 0
            new_row = vcat(new_row, ones(number_new_users))
            activity_matrix = hcat(
                activity_matrix,
                zeros(day - 1, number_new_users),
            )
        end
        activity_matrix = vcat(activity_matrix, new_row')
    end

    return activity_matrix
end


"""Return the paper's new-user predictive law at every follow-up horizon."""
function new_user_distributions(
    pilot_days,
    followup_days,
    observed_users,
    alpha,
    c,
    beta_parameter,
)
    followup_psi = alpha .* cumsum(
        beta.(1 - alpha, pilot_days .+ (1:followup_days)),
    )
    denominator = (
        beta_parameter +
        alpha * sum(beta.(1 - alpha, 1:pilot_days)) .+
        followup_psi
    )
    probabilities = 1.0 .- followup_psi ./ denominator
    return NegativeBinomial.(observed_users + c + 1, probabilities)
end


"""Sample the posterior of the participation hitting times ``D_M``."""
function sample_dm_posterior(
    targets,
    observed_first_trigger_days,
    alpha,
    c,
    beta_parameter,
    pilot_days,
    upper_day,
    number_draws::Integer,
)
    samples = zeros(number_draws, length(targets))
    for iteration in 1:number_draws
        predicted_first_trigger_days = sort(
            sample_tg_ssp_first_trigger_times(
                observed_first_trigger_days,
                alpha,
                c,
                beta_parameter,
                pilot_days,
                upper_day,
            ),
        )

        for (target_index, target) in enumerate(targets)
            if target > length(predicted_first_trigger_days)
                println(
                    "Warning: target ",
                    targets[target_index],
                    " exceeds the users active by upper_day; try a larger upper_day",
                )
                samples[iteration, target_index] = pilot_days + upper_day
            else
                samples[iteration, target_index] = predicted_first_trigger_days[
                    target
                ]
            end
        end
    end
    return samples
end


"""Simulate the global credible band for cumulative distinct users."""
function new_user_credible_band(
    observed_first_trigger_days,
    alpha,
    c,
    beta_parameter,
    pilot_days,
    followup_days,
    number_draws::Integer,
)
    function posterior_scale_distribution()
        shape = c + length(observed_first_trigger_days) + 1
        rate = beta_parameter + alpha * sum(
            beta.(1 - alpha, 1:pilot_days),
        )
        return Gamma(shape, 1.0 / rate)
    end

    function simulate_trajectory()
        scale_distribution = posterior_scale_distribution()
        scale = rand(scale_distribution)
        increment_distributions = Poisson.(
            alpha .* scale .* beta.(
                1 - alpha,
                pilot_days .+ (1:followup_days),
            ),
        )
        new_users_each_day = rand.(increment_distributions)
        trajectory = cumsum(new_users_each_day)
        log_probability = (
            logpdf(scale_distribution, scale) +
            sum(logpdf.(increment_distributions, new_users_each_day))
        )
        return trajectory, log_probability
    end

    trajectories = zeros((number_draws, followup_days))
    log_probabilities = zeros(number_draws)
    for draw in 1:number_draws
        trajectories[draw, :], log_probabilities[draw] = simulate_trajectory()
    end

    cutoff = quantile(log_probabilities, 0.05)
    trajectories = trajectories[log_probabilities .>= cutoff, :]
    upper = maximum(trajectories, dims=1)[1, :]
    lower = minimum(trajectories, dims=1)[1, :]
    observed_users = length(observed_first_trigger_days)
    return observed_users .+ lower, observed_users .+ upper
end
