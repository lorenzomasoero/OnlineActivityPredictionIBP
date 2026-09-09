using Optimization
using OptimizationOptimJL
using SpecialFunctions: beta, logbeta, loggamma


function _be_ssp_log_marginal(activity_matrix, alpha, c, beta_parameter)
    active_day_counts = sum(activity_matrix, dims=1)
    pilot_days, observed_users = size(activity_matrix)
    psi_pilot = alpha * sum(beta.(1 - alpha, 1:pilot_days))

    log_likelihood = (
        observed_users * log(alpha) +
        (c + 1) * log(beta_parameter) -
        (observed_users + c + 1) * log(beta_parameter + psi_pilot)
    )
    log_likelihood += (
        loggamma(observed_users + c + 1) - loggamma(c + 1)
    )
    log_likelihood += sum(
        logbeta.(
            active_day_counts .- alpha,
            pilot_days .- active_day_counts .+ 1,
        ),
    )
    return log_likelihood
end


"""Fit the paper's Bernoulli stable-beta scaled process (Be-SSP)."""
function fit_be_ssp(activity_matrix)
    objective(parameters, _) = -_be_ssp_log_marginal(
        activity_matrix,
        parameters[1],
        parameters[2],
        parameters[3],
    )
    optimization_function = OptimizationFunction(
        objective,
        Optimization.AutoForwardDiff(),
    )
    problem = OptimizationProblem(
        optimization_function,
        [0.1, 10.0, 5.0],
        lb=[0.001, 1e-8, 1e-8],
        ub=[0.999, Inf, 10.0],
    )
    solution = solve(problem, GradientDescent())
    return solution.u
end


function _tg_ssp_log_marginal(
    first_trigger_days,
    alpha,
    c,
    beta_parameter,
    pilot_days,
)
    @assert maximum(first_trigger_days) <= pilot_days
    observed_users = length(first_trigger_days)
    psi_pilot = alpha * sum(beta.(1 - alpha, 1:pilot_days))

    log_likelihood = (
        observed_users * log(alpha) +
        (c + 1) * log(beta_parameter) -
        (observed_users + c + 1) * log(beta_parameter + psi_pilot)
    )
    log_likelihood += (
        loggamma(observed_users + c + 1) - loggamma(c + 1)
    )
    log_likelihood += sum(logbeta.(1 - alpha, first_trigger_days))
    return log_likelihood
end


"""Fit the paper's first-trigger-time stable-beta scaled process (TG-SSP)."""
function fit_tg_ssp(first_trigger_days, pilot_days)
    objective(parameters, _) = -_tg_ssp_log_marginal(
        first_trigger_days,
        parameters[1],
        parameters[2],
        parameters[3],
        pilot_days,
    )
    optimization_function = OptimizationFunction(
        objective,
        Optimization.AutoForwardDiff(),
    )
    problem = OptimizationProblem(
        optimization_function,
        [0.1, 1000.0, 0.5],
        lb=[0.001, 1e-8, 1e-8],
        ub=[0.999, 10000.0, 10.0],
    )
    solution = solve(problem, BFGS())
    return solution.u
end
