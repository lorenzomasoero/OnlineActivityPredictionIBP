using CSV
using DataFrames
using Distributions
using ProgressBars
using Random
using Statistics

include(joinpath(@__DIR__, "BnpActivity.jl"))
using .BnpActivity


const PILOT_DAYS = 14
const NUMBER_POSTERIOR_DRAWS = 1000
const NUMBER_REPETITIONS = 500
const NUMBER_MAX_CUSTOMERS = 500000
const RANDOM_SEED = 20230903
const TARGET_MULTIPLIERS = [1.5, 2.0, 5.0]
const TAIL_PARAMETERS = [0.8, 1.0, 1.2, 1.4]


function _inverse_interval(
    target,
    lower_band,
    upper_band,
    pilot_days,
)
    lower_day = findfirst(upper_band .>= target)
    upper_day = findlast(lower_band .<= target)
    if lower_day === nothing
        lower_day = 1
    end
    if upper_day === nothing
        upper_day = length(lower_band)
    else
        upper_day += 1
    end
    return pilot_days .+ sort([lower_day, upper_day])
end


function _new_user_prediction(
    pilot_days,
    upper_day,
    observed_users,
    alpha,
    c,
    beta_parameter,
)
    predictives = new_user_distributions(
        pilot_days,
        upper_day,
        observed_users,
        alpha,
        c,
        beta_parameter,
    )
    means = observed_users .+ [mean(distribution) for distribution in predictives]
    lower = observed_users .+ [
        quantile(distribution, 0.025) for distribution in predictives
    ]
    upper = observed_users .+ [
        quantile(distribution, 0.975) for distribution in predictives
    ]
    return means, (lower, upper)
end


_posterior_interval!(samples) = quantile!(samples, [0.025, 0.975])


_in_interval(value, lower, upper) = 1.0 * (
    (value >= lower) & (value <= upper)
)


function _sample_zipf_first_trigger_days(
    tail_parameter,
    number_days,
    number_max_customers=NUMBER_MAX_CUSTOMERS,
)
    activity_probabilities = (1:number_max_customers) .^ (-tail_parameter)
    activity = zeros(number_days, number_max_customers)
    for day in 1:number_days
        activity[day, :] .= rand.(Bernoulli.(activity_probabilities))
    end
    first_trigger_days = findfirst.(
        values -> values .> 0,
        eachcol(activity),
    )
    return first_trigger_days[first_trigger_days .!= nothing]
end


function run_simulation(tail_parameter, repetition)
    upper_day = if tail_parameter < 1
        1000
    elseif tail_parameter < 1.4
        50000
    else
        250000
    end

    all_first_trigger_days = sort(
        _sample_zipf_first_trigger_days(tail_parameter, 365),
    )
    pilot_first_trigger_days = all_first_trigger_days[
        all_first_trigger_days .<= PILOT_DAYS
    ]
    observed_users = length(pilot_first_trigger_days)
    targets = Integer.(ceil.(observed_users .* TARGET_MULTIPLIERS))
    true_dm = zeros(length(targets))
    for (target_index, target) in enumerate(targets)
        true_dm[target_index] = all_first_trigger_days[target]
    end

    alpha, c, beta_parameter = fit_tg_ssp(
        pilot_first_trigger_days,
        PILOT_DAYS,
    )

    cumulative_mean, _ = _new_user_prediction(
        PILOT_DAYS,
        upper_day,
        observed_users,
        alpha,
        c,
        beta_parameter,
    )
    cumulative_band = new_user_credible_band(
        pilot_first_trigger_days,
        alpha,
        c,
        beta_parameter,
        PILOT_DAYS,
        upper_day,
        NUMBER_POSTERIOR_DRAWS,
    )

    inverse_estimates = zeros(length(targets))
    for (target_index, target) in enumerate(targets)
        inverse_estimates[target_index] = (
            PILOT_DAYS + findmin(abs.(cumulative_mean .- target))[2]
        )
    end
    inverse_errors = abs.(inverse_estimates - true_dm)

    inverse_bands = [
        _inverse_interval(
            target,
            cumulative_band[1],
            cumulative_band[2],
            PILOT_DAYS,
        ) for target in targets
    ]
    inverse_coverage = [
        _in_interval(value, band[1], band[2]) for
        (value, band) in zip(true_dm, inverse_bands)
    ]
    inverse_lengths = [band[2] - band[1] for band in inverse_bands]

    posterior_dm_samples = sample_dm_posterior(
        targets .- observed_users,
        pilot_first_trigger_days,
        alpha,
        c,
        beta_parameter,
        PILOT_DAYS,
        upper_day,
        NUMBER_POSTERIOR_DRAWS,
    )
    posterior_means = vec(mean(posterior_dm_samples, dims=1))
    posterior_errors = abs.(posterior_means - true_dm)

    posterior_bands = []
    posterior_coverage = zeros(length(targets))
    for target_index in eachindex(targets)
        current_band = _posterior_interval!(
            posterior_dm_samples[:, target_index],
        )
        push!(posterior_bands, current_band)
        posterior_coverage[target_index] = _in_interval(
            true_dm[target_index],
            current_band[1],
            current_band[2],
        )
    end
    posterior_lengths = [band[2] - band[1] for band in posterior_bands]

    return Dict(
        "tail" => tail_parameter,
        "rep" => repetition,
        "inverse_coverage" => inverse_coverage,
        "inverse_length" => inverse_lengths,
        "inverse_error" => inverse_errors,
        "post_error" => posterior_errors,
        "post_coverage" => posterior_coverage,
        "post_length" => posterior_lengths,
    )
end


function main(output_directory::AbstractString)
    mkpath(output_directory)
    Random.seed!(RANDOM_SEED)

    for tail_parameter in TAIL_PARAMETERS
        println("tail: ", tail_parameter)
        results = Vector{Dict}(undef, NUMBER_REPETITIONS)
        Threads.@threads for repetition in ProgressBar(1:NUMBER_REPETITIONS)
            results[repetition] = run_simulation(tail_parameter, repetition)
        end
        final_data = DataFrame(results)
        filename = "zipf_interval_comparison_tail$(tail_parameter).csv"
        CSV.write(joinpath(output_directory, filename), final_data)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error(
        "Usage: julia --project=. interval_comparison.jl OUTPUT_DIRECTORY",
    )
    main(ARGS[1])
end
