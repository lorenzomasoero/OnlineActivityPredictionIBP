functions{
    real trunc_geom_lpmf(int y, real pi, int d) {
        real out;
        if (y == 0) {
            out = d * log(1 - pi);
        } else if (y <= d) {
            out = (y - 1) * log(1 - pi) + log(pi);
        } else {
            out = negative_infinity();
        }
        return out;
    }

    real q0(real alpha, real beta, int d, int dfollow) {
        real out = lgamma(beta + d + dfollow) + lgamma(alpha + beta + d);
        out = out - (
            lgamma(alpha + beta + d + dfollow) + lgamma(beta + d));
        return exp(out);
    }
}

data {
    int<lower=0> Ndays;
    int<lower=0> Nfollow;

    array[Ndays] int users_per_day;
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
    target += - 2.5 * log(alpha + beta);
    target += Ndata * (lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta));
    target += n_trig * lgamma(alpha + 1);
    for (i in 1:Ndays) {
        target += users_per_day[i] * lgamma(beta + (i-1));
        target += -1.0 * users_per_day[i] * lgamma(alpha + beta + i);
    }

    target += n_notrig * lgamma(beta + Ndays);
    target += -1.0 * n_notrig *lgamma(alpha + beta + Ndays);
}


generated quantities {
    array[Nfollow] int n_new_users;
    for (i in 1:Nfollow) {
        n_new_users[i] = n_notrig - binomial_rng(n_notrig, q0(alpha, beta, Ndays, i));
    }
}
