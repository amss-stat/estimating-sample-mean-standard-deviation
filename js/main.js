// --- Configuration ---
const MODEL_CONFIG = {
    s1: {
        beta:     { mu: 'mu_s1_beta_model.onnx',     sigma: 'sigma_s1_beta_model.onnx' },
        lognormal:{ mu: 'mu_s1_lognormal_model.onnx',sigma: 'sigma_s1_lognormal_model.onnx' },
        weibull:  { mu: 'mu_s1_weibull_model.onnx',  sigma: 'sigma_s1_weibull_model.onnx' },
        normal:   {                                 sigma: 'sigma_s1_normal_model.onnx' },
        exp:      { mu: 's1_exp_model.onnx' }
    },
    s2: {
        beta:     { mu: 'mu_s2_beta_model.onnx',     sigma: 'sigma_s2_beta_model.onnx' },
        lognormal:{ mu: 'mu_s2_lognormal_model.onnx',sigma: 'sigma_s2_lognormal_model.onnx' },
        weibull:  { mu: 'mu_s2_weibull_model.onnx',  sigma: 'sigma_s2_weibull_model.onnx' },
        normal:   {                                 sigma: 'sigma_s2_normal_model.onnx' },
        exp:      { mu: 's2_exp_model.onnx' }
    },
    s3: {
        beta:     { mu: 'mu_s3_beta_model.onnx',     sigma: 'sigma_s3_beta_model.onnx' },
        lognormal:{ mu: 'mu_s3_lognormal_model.onnx',sigma: 'sigma_s3_log_model.onnx' },
        weibull:  { mu: 'mu_s3_weibull_model.onnx',  sigma: 'sigma_s3_weibull_model.onnx' },
        normal:   {                                 sigma: 'sigma_s3_normal_model.onnx' },
        exp:      { mu: 's3_exp_model.onnx' }
    }
};

const PREFERENCE_ORDER_DEFAULT = ['Normal', 'Log-Normal', 'Exponential', 'Weibull', 'Beta'];
const PREFERENCE_ORDER_BETA_PRIORITY = ['Beta', 'Normal', 'Log-Normal', 'Exponential', 'Weibull'];

const quantileFunctions = {
    beta: (p, params) => jStat.beta.inv(p, params.alpha, params.beta),
    weibull: (p, params) => params.lambda * Math.pow(-Math.log(1 - p), 1 / params.k),
    lognormal: (p, params) => Math.exp(params.mu_ln + params.sigma_ln * jStat.normal.inv(p, 0, 1)),
    exp: (p, params) => -params.theta * Math.log(1 - p),
    normal: (p, params) => params.mu + params.sigma * jStat.normal.inv(p, 0, 1),
};

/**
 * Numerically solves for the Weibull shape parameter 'k' using the bisection method.
 * It finds the root of the equation f(k) = (sigma/mu)^2 - (Gamma(1+2/k)/Gamma(1+1/k)^2 - 1) = 0.
 * @param {number} mu - The mean of the distribution.
 * @param {number} sigma - The standard deviation of the distribution.
 * @returns {number|null} The solved shape parameter 'k', or null if no solution is found.
 */
function solveForWeibullK(mu, sigma) {
    if (mu <= 0 || sigma <= 0) return null;

    const target_cv_sq = (sigma / mu) ** 2;

    const f = (k) => {
        if (k <= 0) return Infinity;
        const g1 = jStat.gammafn(1 + 1 / k);
        const g2 = jStat.gammafn(1 + 2 / k);
        if (g1 === 0 || isNaN(g1) || isNaN(g2)) return Infinity;
        return (g2 / (g1 ** 2)) - 1 - target_cv_sq;
    };

    let low = 0.1; // Lower bound for k
    let high = 20.0; // Upper bound for k
    const TOLERANCE = 1e-6;
    const MAX_ITERATIONS = 100;

    // Check if the root is bracketed
    if (f(low) * f(high) >= 0) {
        // console.warn("Weibull k solver: root not bracketed. The CV might be out of the typical range for Weibull.");
        // Fallback to the (less reliable) approximation for extreme cases
        return Math.pow(sigma / mu, -1.086);
    }

    let k_est = (low + high) / 2;
    for (let i = 0; i < MAX_ITERATIONS; i++) {
        k_est = (low + high) / 2;
        const y_mid = f(k_est);

        if (Math.abs(y_mid) < TOLERANCE) {
            return k_est; // Solution found
        }

        if (f(low) * y_mid < 0) {
            high = k_est;
        } else {
            low = k_est;
        }
    }
    return k_est; // Return best estimate after max iterations
}

// --- Global State ---
const modelSessions = {};
let currentScenario = 's1';
const ui = {
    button: document.getElementById('predict-button'),
    status: document.getElementById('status-message'),
    resultContainer: document.getElementById('result-container'),
    resultText: document.getElementById('result-text'),
    inputs: {
        n: document.getElementById('n'), m: document.getElementById('m'),
        a: document.getElementById('a'), b: document.getElementById('b'),
        q1: document.getElementById('q1'), q3: document.getElementById('q3'),
    },
    groups: {
        a: document.getElementById('group-a'), b: document.getElementById('group-b'),
        q1: document.getElementById('group-q1'), q3: document.getElementById('group-q3'),
    },
    subtitle: document.getElementById('subtitle'),
    scenarioRadios: document.querySelectorAll('input[name="scenario"]')
};

// --- Model Loading ---
async function loadModel(name, path) {
    try {
        const session = await ort.InferenceSession.create(path);
        modelSessions[name] = session;
    } catch (e) {
        console.error(`Failed to load model ${name} from ${path}`, e);
        throw e;
    }
}

async function initializeModels() {
    ui.status.textContent = '⏳ Loading all models, please wait...';
    ui.button.disabled = true;
    try {
        const loadPromises = [];
        for (const scenario in MODEL_CONFIG) {
            for (const dist in MODEL_CONFIG[scenario]) {
                for (const param in MODEL_CONFIG[scenario][dist]) {
                    const uniqueName = `${scenario}_${dist}_${param}`;
                    const path = `models/${MODEL_CONFIG[scenario][dist][param]}`;
                    loadPromises.push(loadModel(uniqueName, path));
                }
            }
        }
        await Promise.all(loadPromises);
        ui.status.textContent = '✅ Models loaded. Please select a scenario and enter your data.';
        ui.button.disabled = false;
    } catch (error) {
        ui.status.textContent = '❌ Failed to load models. Check console and refresh.';
    }
}

// --- UI Management ---
function updateUIForScenario(scenario) {
    currentScenario = scenario;
    const isS1 = scenario === 's1';
    const isS2 = scenario === 's2';

    ui.groups.a.classList.toggle('hidden', isS2);
    ui.groups.b.classList.toggle('hidden', isS2);
    ui.groups.q1.classList.toggle('hidden', isS1);
    ui.groups.q3.classList.toggle('hidden', isS1);

    if (isS1) ui.subtitle.textContent = 'From Sample Size, Minimum, Median, and Maximum';
    if (isS2) ui.subtitle.textContent = 'From Sample Size, First Quartile, Median, and Third Quartile';
    if (scenario === 's3') ui.subtitle.textContent = 'From All Available Statistics';
    
    ui.resultContainer.classList.add('hidden');
    ui.status.textContent = 'Please enter new values for the selected scenario.';
}

// --- Core Calculation Logic ---
async function runInference(session, features, inputShape) {
    const tensor = new ort.Tensor('float32', new Float32Array(features), inputShape);
    const feeds = { [session.inputNames[0]]: tensor };
    const results = await session.run(feeds);
    return results[session.outputNames[0]].data[0];
}

async function calculateLoss(distParams, features, scenario) {
    const { n, a, q1, m, q3, b } = features;
    const quantileFn = quantileFunctions[distParams.distName.toLowerCase()];
    if (typeof quantileFn !== 'function') {
        console.error(`Could not find quantile function for distribution: ${distParams.distName}`);
        return Infinity;
    }

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
    } else { // scenario === 's3'
        const q_min = quantileFn(1 / n, distParams.params);
        const q25 = quantileFn(0.25, distParams.params);
        const q50 = quantileFn(0.5, distParams.params);
        const q75 = quantileFn(0.75, distParams.params);
        const q_max = quantileFn(1 - 1 / n, distParams.params);
        loss = Math.pow(q_min - a, 2) + Math.pow(q25 - q1, 2) + Math.pow(q50 - m, 2) + Math.pow(q75 - q3, 2) + Math.pow(q_max - b, 2);
    }
    return isNaN(loss) ? Infinity : loss;
}

async function handlePrediction() {
    const inputs = {};
    for (const key in ui.inputs) {
        inputs[key] = parseFloat(ui.inputs[key].value);
    }
    const { n: n_original, m, a, b, q1, q3 } = inputs;
    
    const requiredInputs = currentScenario === 's1' ? [n_original, m, a, b] :
                           currentScenario === 's2' ? [n_original, m, q1, q3] :
                           [n_original, m, a, b, q1, q3];
    if (requiredInputs.some(isNaN)) {
        alert("Please enter valid numbers for all fields required by the scenario.");
        return;
    }
    const validationMap = { s1: a <= m && m <= b, s2: q1 <= m && m <= q3, s3: a <= q1 && q1 <= m && m <= q3 && q3 <= b };
    if (!validationMap[currentScenario]) {
        alert("Input validation failed: values are not in logical order (e.g., min <= median <= max).");
        return;
    }

    const n = Math.min(n_original, 1000);
    ui.button.disabled = true;
    ui.status.textContent = '🧠 Calculating...';
    ui.resultContainer.classList.add('hidden');

    try {
        const allFeatures = { n, m, a, b, q1, q3 };
        const results = [];
        let distributions = ['beta', 'weibull', 'lognormal', 'normal', 'exp'];

        // --- NEW: Extreme Asymmetry Rule to Exclude Normal Distribution ---
        const ASYMMETRY_RATIO_THRESHOLD = 10; // e.g., one side is 10x longer than the other
        let excludeNormalDueToAsymmetry = false;
        
        // Note: For S3, if either min/max OR quartiles are extremely asymmetric, we exclude Normal.
        if (currentScenario === 's1' || currentScenario === 's3') {
            const left = m - a;
            const right = b - m;
            if (left > 0 && right > 0 && (Math.max(left, right) / Math.min(left, right) > ASYMMETRY_RATIO_THRESHOLD)) {
                excludeNormalDueToAsymmetry = true;
            }
        }
        if (currentScenario === 's2' || currentScenario === 's3') {
            const left = m - q1;
            const right = q3 - m;
             if (left > 0 && right > 0 && (Math.max(left, right) / Math.min(left, right) > ASYMMETRY_RATIO_THRESHOLD)) {
                excludeNormalDueToAsymmetry = true;
            }
        }
        
        if (excludeNormalDueToAsymmetry) {
            distributions = distributions.filter(d => d !== 'normal');
        }
        // --- End of New Rule ---

        const scalingFactors = { normal: 1, exp: 1 };
        
        let roughSigma = 0;
        if (currentScenario === 's1') {
            const range = b - a;
            if (range > 0) roughSigma = range / 4;
        } else {
            const iqr = q3 - q1;
            if (iqr > 0) roughSigma = iqr / 1.349;
        }
        if (roughSigma > 0 && (roughSigma < 5 || roughSigma > 15)) {
            scalingFactors.normal = 10 / roughSigma;
        }

        if ((currentScenario === 's1' || currentScenario === 's3') && m > 0 && m < 1) {
            scalingFactors.exp = 10 / m;
        }

        for (const dist of distributions) {
            let mu, sigma, params, distName = dist.charAt(0).toUpperCase() + dist.slice(1);
            
            const scaleFactor = scalingFactors[dist] || 1;
            let scaledFeatures = allFeatures;

            if (scaleFactor !== 1) {
                scaledFeatures = {
                    n, m: m * scaleFactor, a: a * scaleFactor, b: b * scaleFactor,
                    q1: q1 * scaleFactor, q3: q3 * scaleFactor
                };
            }

            if (dist === 'normal') {
                
                if (currentScenario === 's1') {
                    const n_pow_075 = Math.pow(n, 0.75);
                    const w1 = 4 / (4 + n_pow_075);
                    const w2 = 1 - w1; // Weights sum to 1
                    const mid_point = (a + b) / 2;
                    mu = w1 * mid_point + w2 * m;
                } else if (currentScenario === 's2') {
                    const w1 = 0.7 + 0.39 / n;
                    const w2 = 1 - w1; // Weights sum to 1
                    const mid_quartile = (q1 + q3) / 2;
                    mu = w1 * mid_quartile + w2 * m;
                } else { // Scenario 3
                    const n_pow_075 = Math.pow(n, 0.75);
                    const n_pow_055 = Math.pow(n, 0.55);
                    const w1 = 2.2 / (2.2 + n_pow_075);
                    const w2 = 0.7 - (0.72 / n_pow_055);
                    const w3 = 1 - w1 - w2; // Weights sum to 1
                    const mid_point = (a + b) / 2;
                    const mid_quartile = (q1 + q3) / 2;
                    mu = w1 * mid_point + w2 * mid_quartile + w3 * m;
                }
                
                let features, shape;
                if (currentScenario === 's1') { features = [scaledFeatures.n, scaledFeatures.a - scaledFeatures.m, scaledFeatures.b - scaledFeatures.m]; shape = [1, 3]; }
                if (currentScenario === 's2') { features = [scaledFeatures.n, scaledFeatures.q1 - scaledFeatures.m, scaledFeatures.q3 - scaledFeatures.m]; shape = [1, 3]; }
                if (currentScenario === 's3') { features = [scaledFeatures.n, scaledFeatures.a - scaledFeatures.m, scaledFeatures.q1 - scaledFeatures.m, scaledFeatures.q3 - scaledFeatures.m, scaledFeatures.b - scaledFeatures.m]; shape = [1, 5]; }
                
                let sigma_scaled = await runInference(modelSessions[`${currentScenario}_normal_sigma`], features, shape);
                sigma = sigma_scaled / scaleFactor;
                params = { mu, sigma };
            } else {
                let features, shape;
                if (currentScenario === 's1') { features = [scaledFeatures.n, scaledFeatures.a, scaledFeatures.m, scaledFeatures.b]; shape = [1, 4]; }
                if (currentScenario === 's2') { features = [scaledFeatures.n, scaledFeatures.q1, scaledFeatures.m, scaledFeatures.q3]; shape = [1, 4]; }
                if (currentScenario === 's3') { features = [scaledFeatures.n, scaledFeatures.a, scaledFeatures.q1, scaledFeatures.m, scaledFeatures.q3, scaledFeatures.b]; shape = [1, 6]; }

                if (!MODEL_CONFIG[currentScenario][dist]) continue;

                const supportVal1 = currentScenario === 's1' || currentScenario === 's3' ? a : q1;
                const supportVal2 = currentScenario === 's1' || currentScenario === 's3' ? b : q3;
                if (dist === 'beta' && (supportVal1 < 0 || supportVal2 > 1)) continue;
                if (['weibull', 'lognormal', 'exp'].includes(dist) && supportVal1 <= 0) continue;

                let mu_scaled = await runInference(modelSessions[`${currentScenario}_${dist}_mu`], features, shape);
                let sigma_scaled = MODEL_CONFIG[currentScenario][dist].sigma 
                      ? await runInference(modelSessions[`${currentScenario}_${dist}_sigma`], features, shape) 
                      : mu_scaled;

                mu = mu_scaled / scaleFactor;
                sigma = sigma_scaled / scaleFactor;
                
                if (dist === 'beta') {
                    const sum = mu * (1 - mu) / (sigma ** 2) - 1;
                    params = sum > 0 ? { alpha: sum * mu, beta: sum * (1 - mu) } : null;
                } else if (dist === 'weibull') {
                    // --- ★★★ FIX APPLIED HERE ★★★ ---
                    // Replace the old approximation with the new numerical solver
                    const k = solveForWeibullK(mu, sigma);
                    if (k === null || isNaN(k)) {
                        params = null;
                    } else {
                        const lambda = mu / jStat.gammafn(1 + 1 / k);
                        params = { k, lambda };
                    };
                } else if (dist === 'lognormal') {
                    const sigma_ln_sq = Math.log(1 + (sigma / mu) ** 2);
                    params = (mu > 0 && sigma_ln_sq >= 0) ? { mu_ln: Math.log(mu) - sigma_ln_sq / 2, sigma_ln: Math.sqrt(sigma_ln_sq) } : null;
                } else if (dist === 'exp') {
                    params = { theta: sigma };
                }
            }
            
            if (params && !Object.values(params).some(isNaN)) {
                const loss = await calculateLoss({ distName, params }, allFeatures, currentScenario);
                if (isFinite(loss)) {
                    results.push({ name: distName, loss, mean: mu, std: sigma, params });
                }
            }
        }
        
        if (results.length === 0) {
            ui.status.textContent = '❌ No suitable distribution found. The data may fall outside the support of the tested distributions.';
            return;
        }
        
        results.sort((x, y) => x.loss - y.loss);
        
        let bestFit = results[0];
        let final_decision_made = false;

        // 1. Domain Priority Rule: Check for Beta domain first.
        const TIE_BREAKING_RATIO = 1.05;
        let isBetaDomain = false;
        if (currentScenario === 's1' || currentScenario === 's3') {
            isBetaDomain = (a >= 0 && b <= 1);
        } else { // currentScenario === 's2'
            isBetaDomain = (q1 >= 0 && q3 <= 1);
        }

        if (isBetaDomain) {
            const betaResult = results.find(r => r.name === 'Beta');
            if (betaResult && (betaResult.loss / results[0].loss) < TIE_BREAKING_RATIO) {
                bestFit = betaResult;
                final_decision_made = true;
            }
        }

        // 2. Symmetry Override Rule: If no decision yet, check for strict symmetry.
        if (!final_decision_made) {
            const SYMMETRY_TOLERANCE = 0.001;
            let isStrictlySymmetric = false;

            if (currentScenario === 's1') {
                const left = m - a, right = b - m;
                if (left > 0 && right > 0 && Math.abs(left - right) / (left + right) < SYMMETRY_TOLERANCE) {
                    isStrictlySymmetric = true;
                }
            } else if (currentScenario === 's2') {
                const left = m - q1, right = q3 - m;
                if (left > 0 && right > 0 && Math.abs(left - right) / (left + right) < SYMMETRY_TOLERANCE) {
                    isStrictlySymmetric = true;
                }
            } else { // Scenario 3
                const left1 = m - a, right1 = b - m;
                const left2 = m - q1, right2 = q3 - m;
                const s1_sym = (left1 > 0 && right1 > 0 && Math.abs(left1 - right1) / (left1 + right1) < SYMMETRY_TOLERANCE);
                const s2_sym = (left2 > 0 && right2 > 0 && Math.abs(left2 - right2) / (left2 + right2) < SYMMETRY_TOLERANCE);
                if (s1_sym && s2_sym) {
                    isStrictlySymmetric = true;
                }
            }
            
            const normalResult = results.find(r => r.name === 'Normal');
            if (isStrictlySymmetric && normalResult) {
                bestFit = normalResult;
                final_decision_made = true;
            }
        }

        // 3. Default Tie-Breaking: If still no decision, apply default preference.
        if (!final_decision_made) {
            const bestLoss = results[0].loss;
            for (const preferredDistName of PREFERENCE_ORDER_DEFAULT) {
                const preferredResult = results.find(r => r.name === preferredDistName);
                if (preferredResult && (preferredResult.loss / bestLoss) < TIE_BREAKING_RATIO) {
                    bestFit = preferredResult;
                    break;
                }
            }
        }
        
        let output = `Best Fit Distribution: ${bestFit.name}\n`;
        output += `-------------------------------------------\n`;
        output += `Estimated Sample Mean : ${bestFit.mean.toFixed(4)}\n`;
        output += `Estimated Sample SD   : ${bestFit.std.toFixed(4)}\n\n`;
        output += `Best Fit Distribution Parameters:\n`;
        const paramsString = Object.entries(bestFit.params).map(([key, value]) => `  ${key.padEnd(7)}: ${value.toFixed(4)}`).join('\n');
        output += paramsString;
        
        ui.resultText.textContent = output;
        ui.resultContainer.classList.remove('hidden');
        ui.status.textContent = '✅ Calculation complete. You can enter new data.';
        
    } catch (error) {
        console.error("An error occurred during prediction:", error);
        ui.status.textContent = '❌ An unexpected error occurred. Please check the console for details.';
    } finally {
        ui.button.disabled = false;
    }
}

// --- Event Listeners ---
window.addEventListener('DOMContentLoaded', () => {
    initializeModels();
    updateUIForScenario('s1');
});
ui.button.addEventListener('click', handlePrediction);
ui.scenarioRadios.forEach(radio => radio.addEventListener('change', e => updateUIForScenario(e.target.value)));




