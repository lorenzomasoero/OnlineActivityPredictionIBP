using CSV
using DataFrames
using Distributions
using Random
using Statistics: mean

include(joinpath(@__DIR__, "BnpActivity.jl"))
using .BnpActivity


const PILOT_DAYS = 14
const FOLLOWUP_DAYS = 14
const C = 2500
const BETA = 0.5
const NUMBER_REPETITIONS = 50
const RANDOM_SEED = 20230903


function experiment_from_be_ssp()
    true_alpha = rand(Beta(4, 10))
    all_activity = sample_be_ssp(
        true_alpha,
        C,
        BETA,
        PILOT_DAYS + FOLLOWUP_DAYS,
    )
    println("all_activity: ", size(all_activity))

    pilot_activity = all_activity[1:PILOT_DAYS, :]
    observed_users = findlast(vec(sum(pilot_activity, dims=1)) .> 0)
    pilot_activity = pilot_activity[:, 1:observed_users]

    be_parameters = fit_be_ssp(pilot_activity)
    alpha, c, beta_parameter = be_parameters
    be_ssp_prediction = observed_users + mean(
        new_user_distributions(
            PILOT_DAYS,
            FOLLOWUP_DAYS,
            observed_users,
            alpha,
            c,
            beta_parameter,
        )[end],
    )

    first_trigger_days = findfirst.(
        values -> values .> 0,
        eachcol(pilot_activity),
    )
    tg_parameters = fit_tg_ssp(first_trigger_days, PILOT_DAYS)
    alpha, c, beta_parameter = tg_parameters
    tg_ssp_prediction = observed_users + mean(
        new_user_distributions(
            PILOT_DAYS,
            FOLLOWUP_DAYS,
            observed_users,
            alpha,
            c,
            beta_parameter,
        )[end],
    )

    oracle_prediction = observed_users + mean(
        new_user_distributions(
            PILOT_DAYS,
            FOLLOWUP_DAYS,
            observed_users,
            true_alpha,
            C,
            BETA,
        )[end],
    )
    return (
        true_alpha,
        be_ssp_prediction,
        tg_ssp_prediction,
        oracle_prediction,
        size(all_activity, 2),
    )
end


function experiment_from_tg_ssp()
    true_alpha = rand(Beta(4, 10))
    all_first_trigger_days = sample_tg_ssp_first_trigger_times(
        Int[],
        true_alpha,
        C,
        BETA,
        0,
        PILOT_DAYS + FOLLOWUP_DAYS,
    )
    println(length(all_first_trigger_days))
    first_trigger_days = all_first_trigger_days[
        all_first_trigger_days .<= PILOT_DAYS
    ]

    pilot_activity = zeros(PILOT_DAYS, length(first_trigger_days))
    observed_users = length(first_trigger_days)
    for (user, first_day) in enumerate(first_trigger_days)
        pilot_activity[first_day, user] = 1.0
        activity_probability = (
            (1 - true_alpha) /
            (1 - true_alpha + first_day) *
            rand(Uniform(0.0, 0.5))
        )
        pilot_activity[(first_day + 1):PILOT_DAYS, user] .= rand(
            Bernoulli(activity_probability),
            PILOT_DAYS - first_day,
        )
    end

    be_parameters = fit_be_ssp(pilot_activity)
    alpha, c, beta_parameter = be_parameters
    be_ssp_prediction = observed_users + mean(
        new_user_distributions(
            PILOT_DAYS,
            FOLLOWUP_DAYS,
            observed_users,
            alpha,
            c,
            beta_parameter,
        )[end],
    )

    tg_parameters = fit_tg_ssp(first_trigger_days, PILOT_DAYS)
    alpha, c, beta_parameter = tg_parameters
    tg_ssp_prediction = observed_users + mean(
        new_user_distributions(
            PILOT_DAYS,
            FOLLOWUP_DAYS,
            observed_users,
            alpha,
            c,
            beta_parameter,
        )[end],
    )

    oracle_prediction = observed_users + mean(
        new_user_distributions(
            PILOT_DAYS,
            FOLLOWUP_DAYS,
            observed_users,
            true_alpha,
            C,
            BETA,
        )[end],
    )
    return (
        true_alpha,
        be_ssp_prediction,
        tg_ssp_prediction,
        oracle_prediction,
        length(all_first_trigger_days),
    )
end


function run_repetitions(experiment)
    alphas = zeros(NUMBER_REPETITIONS)
    be_ssp_predictions = zeros(NUMBER_REPETITIONS)
    tg_ssp_predictions = zeros(NUMBER_REPETITIONS)
    oracle_predictions = zeros(NUMBER_REPETITIONS)
    true_user_counts = zeros(NUMBER_REPETITIONS)

    Random.seed!(RANDOM_SEED)
    Threads.@threads for repetition in 1:NUMBER_REPETITIONS
        result = experiment()
        alphas[repetition] = result[1]
        be_ssp_predictions[repetition] = result[2]
        tg_ssp_predictions[repetition] = result[3]
        oracle_predictions[repetition] = result[4]
        true_user_counts[repetition] = result[5]
    end

    # The paper-aligned column names replace the legacy "bern"/"geom" labels.
    return DataFrame(
        alpha=alphas,
        be_ssp=be_ssp_predictions,
        tg_ssp=tg_ssp_predictions,
        oracle=oracle_predictions,
        true_n=true_user_counts,
    )
end


function main(output_directory::AbstractString)
    mkpath(output_directory)

    be_ssp_data = run_repetitions(experiment_from_be_ssp)
    CSV.write(
        joinpath(output_directory, "comparison_data_from_bern.csv"),
        be_ssp_data,
    )

    tg_ssp_data = run_repetitions(experiment_from_tg_ssp)
    CSV.write(
        joinpath(output_directory, "comparison_data_from_geom.csv"),
        tg_ssp_data,
    )
end


if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error(
        "Usage: julia --project=. model_comparison.jl OUTPUT_DIRECTORY",
    )
    main(ARGS[1])
end
