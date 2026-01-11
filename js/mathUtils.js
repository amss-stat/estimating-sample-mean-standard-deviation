export class InvalidInputError extends Error {
    constructor(message) {
        super(message);
        this.name = "InvalidInputError";
    }
}

export const quantileFunctions = {
    beta: (p, params) => jStat.beta.inv(p, params.alpha, params.beta),
    weibull: (p, params) => params.lambda * Math.pow(-Math.log(1 - p), 1 / params.k),
    lognormal: (p, params) => Math.exp(params.mu_ln + params.sigma_ln * jStat.normal.inv(p, 0, 1)),
    exp: (p, params) => -params.theta * Math.log(1 - p),
    normal: (p, params) => params.mu + params.sigma * jStat.normal.inv(p, 0, 1),
};

export function solveForWeibullK(mu, sigma) {
    if (mu <= 0 || sigma <= 0) return null;
    const target_cv_sq = (sigma / mu) ** 2;
    const f = (k) => {
        if (k <= 0) return Infinity;
        const g1 = jStat.gammafn(1 + 1 / k);
        const g2 = jStat.gammafn(1 + 2 / k);
        if (g1 === 0 || isNaN(g1) || isNaN(g2)) return Infinity;
        return (g2 / (g1 ** 2)) - 1 - target_cv_sq;
    };
    let low = 0.1, high = 20.0, TOLERANCE = 1e-6, MAX_ITERATIONS = 100;
    if (f(low) * f(high) >= 0) return Math.pow(sigma / mu, -1.086);
    let k_est = (low + high) / 2;
    for (let i = 0; i < MAX_ITERATIONS; i++) {
        k_est = (low + high) / 2;
        const y_mid = f(k_est);
        if (Math.abs(y_mid) < TOLERANCE) return k_est;
        if (f(low) * y_mid < 0) high = k_est;
        else low = k_est;
    }
    return k_est;
}

export async function calculateLoss(distParams, features, scenario) {
    const { n, a, q1, m, q3, b } = features;
    const quantileFn = quantileFunctions[distParams.name.toLowerCase()];
    if (typeof quantileFn !== 'function') return Infinity;

    let loss;
    if (scenario === 's1') {
        const q_min = quantileFn(1 / n, distParams.params);
        const q_med = quantileFn(0.5, distParams.params);
        const q_max = quantileFn(1 - 1 / n, distParams.params);
        loss = Math.pow(q_min - a, 2) + Math.pow(q_med - m, 2) + Math.pow(q_max - b, 2);
    } else if (scenario === 's2') {
        const q25 = quantileFn(0.25, distParams.params);
        const q50 = quantileFn(0.5, distParams.params);
        const q75 = quantileFn(0.75, distParams.params);
        loss = Math.pow(q25 - q1, 2) + Math.pow(q50 - m, 2) + Math.pow(q75 - q3, 2);
    } else { // s3
        const q_min = quantileFn(1 / n, distParams.params);
        const q25 = quantileFn(0.25, distParams.params);
        const q50 = quantileFn(0.5, distParams.params);
        const q75 = quantileFn(0.75, distParams.params);
        const q_max = quantileFn(1 - 1 / n, distParams.params);
        loss = Math.pow(q_min - a, 2) + Math.pow(q25 - q1, 2) + Math.pow(q50 - m, 2) + Math.pow(q75 - q3, 2) + Math.pow(q_max - b, 2);
    }
    return isNaN(loss) ? Infinity : loss;
}
