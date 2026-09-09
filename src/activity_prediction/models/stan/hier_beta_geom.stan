functions {
    real q0(real alpha, real beta, int observed_days, int followup_days) {
        real log_probability = lgamma(beta + observed_days + followup_days)
            + lgamma(alpha + beta + observed_days)
            - lgamma(alpha + beta + observed_days + followup_days)
            - lgamma(beta + observed_days);
        return exp(log_probability);
    }
}

data {
    int<lower=1> Ndays;
    int<lower=1> Nfollow;
    array[Ndays] int<lower=0> users_per_day;
    int<lower=0> n_notrig;
}

transformed data {
    int n_trig = sum(users_per_day);
    int Ndata = n_trig + n_notrig;
}

parameters {
    real<lower=0> alpha;
    real<lower=0> beta;
}

model {
    target += -2.5 * log(alpha + beta);
    target += Ndata * (lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta));
    target += n_trig * lgamma(alpha + 1);
    for (day in 1:Ndays) {
        target += users_per_day[day] * lgamma(beta + day - 1);
        target += -users_per_day[day] * lgamma(alpha + beta + day);
    }
    target += n_notrig * lgamma(beta + Ndays);
    target += -n_notrig * lgamma(alpha + beta + Ndays);
}

generated quantities {
    array[Nfollow] int n_new_users;
    for (day in 1:Nfollow) {
        n_new_users[day] = n_notrig
            - binomial_rng(n_notrig, q0(alpha, beta, Ndays, day));
    }
}
