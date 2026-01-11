import { MODEL_CONFIG, PREFERENCE_ORDER_DEFAULT } from './config.js';
import { InvalidInputError, quantileFunctions, solveForWeibullK, calculateLoss } from './mathUtils.js';

const modelSessions = {};

async function loadModel(name, path) {
    try {
        const session = await ort.InferenceSession.create(path);
        modelSessions[name] = session;
    } catch (e) {
        console.error(`Failed to load model ${name} from ${path}`, e);
        throw e;
    }
}

export async function initializeModels() {
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
}

async function runInference(session, features, inputShape) {
    const tensor = new ort.Tensor('float32', new Float32Array(features), inputShape);
    const feeds = { [session.inputNames[0]]: tensor };
    const results = await session.run(feeds);
    return results[session.outputNames[0]].data[0];
}


async function getDistributionResult(dist, currentScenario, allFeatures) {
    const { n, m, a, b, q1, q3 } = allFeatures;
    let mu, sigma, params, distName = dist.charAt(0).toUpperCase() + dist.slice(1);
    
    const COMFORT_ZONES = {
        normal: { target: 20, range: [5, 25] },
        exp: { target: 10, range: [5, 20] },
        lognormal: { target: 300, range: [10, 500] },
        weibull: { target: 20, range: [5, 40] }
    };
    
    let roughSigma = 0;
    if (currentScenario === 's1' || currentScenario === 's3') roughSigma = (b - a) / 4;
    else roughSigma = (q3 - q1) / 1.349;

    let scaleFactor = 1;
    const zone = COMFORT_ZONES[dist];
    if (zone && roughSigma > 0 && (roughSigma < zone.range[0] || roughSigma > zone.range[1])) {
        scaleFactor = zone.target / roughSigma;
    }
    
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
            mu = (4 / (4 + n_pow_075)) * ((a + b) / 2) + (1 - 4 / (4 + n_pow_075)) * m;
        } else if (currentScenario === 's2') {
            mu = (0.7 + 0.39 / n) * ((q1 + q3) / 2) + (1 - (0.7 + 0.39 / n)) * m;
        } else {
            const n_pow_075 = Math.pow(n, 0.75), n_pow_055 = Math.pow(n, 0.55);
            const w1 = 2.2 / (2.2 + n_pow_075), w2 = 0.7 - (0.72 / n_pow_055), w3 = 1 - w1 - w2;
            mu = w1 * ((a + b) / 2) + w2 * ((q1 + q3) / 2) + w3 * m;
        }

        let features, shape;
        if (currentScenario === 's1') { features = [scaledFeatures.n, scaledFeatures.a - scaledFeatures.m, scaledFeatures.b - scaledFeatures.m]; shape = [1, 3]; }
        else if (currentScenario === 's2') { features = [scaledFeatures.n, scaledFeatures.q1 - scaledFeatures.m, scaledFeatures.q3 - scaledFeatures.m]; shape = [1, 3]; }
        else { features = [scaledFeatures.n, scaledFeatures.a - scaledFeatures.m, scaledFeatures.q1 - scaledFeatures.m, scaledFeatures.q3 - scaledFeatures.m, scaledFeatures.b - scaledFeatures.m]; shape = [1, 5]; }
        
        sigma = await runInference(modelSessions[`${currentScenario}_normal_sigma`], features, shape) / scaleFactor;
        params = { mu, sigma };

    } else {
        let features, shape;
        if (currentScenario === 's1') { features = [scaledFeatures.n, scaledFeatures.a, scaledFeatures.m, scaledFeatures.b]; shape = [1, 4]; }
        else if (currentScenario === 's2') { features = [scaledFeatures.n, scaledFeatures.q1, scaledFeatures.m, scaledFeatures.q3]; shape = [1, 4]; }
        else { features = [scaledFeatures.n, scaledFeatures.a, scaledFeatures.q1, scaledFeatures.m, scaledFeatures.q3, scaledFeatures.b]; shape = [1, 6]; }

        mu = await runInference(modelSessions[`${currentScenario}_${dist}_mu`], features, shape) / scaleFactor;
        sigma = MODEL_CONFIG[currentScenario][dist].sigma 
              ? await runInference(modelSessions[`${currentScenario}_${dist}_sigma`], features, shape) / scaleFactor
              : mu;

        if (dist === 'beta') {
            const sum = mu * (1 - mu) / (sigma ** 2) - 1;
            const alpha = sum * mu, beta = sum * (1 - mu);
            params = (alpha > 0 && beta > 0) ? { alpha, beta } : null;
        } else if (dist === 'weibull') {
            const k = solveForWeibullK(mu, sigma);
            if (k > 0) {
                const lambda = mu / jStat.gammafn(1 + 1 / k);
                params = (lambda > 0) ? { k, lambda } : null;
            } else { params = null; }
        } else if (dist === 'lognormal') {
            const sigma_ln_sq = Math.log(1 + (sigma / mu) ** 2);
            if (mu > 0 && sigma_ln_sq >= 0) {
                const sigma_ln = Math.sqrt(sigma_ln_sq);
                params = { mu_ln: Math.log(mu) - sigma_ln_sq / 2, sigma_ln };
            } else { params = null; }
        } else if (dist === 'exp') {
            params = (sigma > 0) ? { theta: sigma } : null;
        }
    }

    if (!params) return null;
    return { name: distName, mean: mu, std: sigma, params };
}

export async function calculateBestDistribution(currentScenario, inputs) {
    const { n: n_original, m, a, b, q1, q3 } = inputs;
    const n = Math.min(n_original, 1000);
    const allFeatures = { n, m, a, b, q1, q3 };
    const warnings = [];
    
    // ★★★ ASYMMETRY LOGIC - REVISED AND CORRECTED ★★★
    let left, right;
    // Prioritize stable quartiles (S2, S3) over min/max (S1) for asymmetry checks
    if (currentScenario === 's2' || currentScenario === 's3') {
        left = m - q1;
        right = q3 - m;
    } else { // S1 only has min/max
        left = m - a;
        right = b - m;
    }

    let excludeNormal = false;
    if (left > 0 && right > 0) {
        const ratio = Math.max(left, right) / Math.min(left, right);
        const relativeDiff = Math.abs(left - right) / (left + right);

        if (ratio > 20) {
            throw new InvalidInputError("Extreme asymmetry detected. Data may contain outliers, making estimation unreliable.");
        }
        
        if (relativeDiff < 0.01) { // 1% tolerance for strict symmetry
            const result = await getDistributionResult('normal', currentScenario, allFeatures);
            if (!result) throw new Error("Symmetric data failed Normal distribution calculation.");
            return { bestFit: result, warnings: ["Data is strictly symmetric; Normal distribution was directly selected."] };
        }
        
        if (ratio >= 2) {
            excludeNormal = true;
        }
    }

    // --- Heuristic Rules Application ---
    let candidates = ['beta', 'weibull', 'lognormal', 'normal', 'exp'];

    const isBetaDomain = (currentScenario === 's1' || currentScenario === 's3') ? (a >= 0 && b <= 1) : (q1 >= 0 && q3 <= 1);
    if (isBetaDomain) {
        const result = await getDistributionResult('beta', currentScenario, allFeatures);
        if (!result) throw new Error("Data in [0,1] failed Beta distribution calculation.");
        return { bestFit: result, warnings: ["Data is in [0,1] range; Beta distribution was directly selected."] };
    } else {
        candidates = candidates.filter(d => d !== 'beta');
    }
    
    if (excludeNormal) {
        candidates = candidates.filter(d => d !== 'normal');
    }
    
    // ★★★ EXPONENTIAL CHECK - REVISED AND CORRECTED ★★★
    if (n > 100) {
        if (currentScenario === 's2' || currentScenario === 's3') {
            const THEORETICAL_RATIO = 0.5847; // Math.log(4/3) / Math.log(2)
            const observedRatio = (m - q1) / (q3 - m);
            const relativeError = Math.abs(observedRatio - THEORETICAL_RATIO) / THEORETICAL_RATIO;
            
            // If the observed ratio is very close to the theoretical one
            if (relativeError < 0.01) { // 1% tolerance
                const result = await getDistributionResult('exp', currentScenario, allFeatures);
                if (!result) throw new Error("Data with memoryless property failed Exponential calculation.");
                return { bestFit: result, warnings: ["Data exhibits strong memoryless property; Exponential distribution was directly selected."] };
            }
            // If the observed ratio is very far from the theoretical one
            if (relativeError > 2) { // 200% tolerance for exclusion
                 candidates = candidates.filter(d => d !== 'exp');
            }
        }
    }

    const results = [];
    for (const dist of candidates) {
        const result = await getDistributionResult(dist, currentScenario, allFeatures);
        if (result) {
            const loss = await calculateLoss(result, allFeatures, currentScenario);
            if (isFinite(loss)) {
                results.push({ ...result, loss });
            }
        }
    }

    if (results.length === 0) {
        throw new Error("No suitable distribution could be fitted after applying heuristic rules.");
    }

    results.sort((a, b) => a.loss - b.loss);
    
    let bestFit = results[0];
    const TIE_BREAKING_RATIO = 1.001;
    const bestLoss = results[0].loss;

    for (const preferredDistName of PREFERENCE_ORDER_DEFAULT) {
        const preferredResult = results.find(r => r.name === preferredDistName);
        if (preferredResult && (preferredResult.loss / bestLoss) < TIE_BREAKING_RATIO) {
            bestFit = preferredResult;
            break;
        }
    }
    
    const numPoints = (currentScenario === 's3') ? 5 : 3;
    const rmse = Math.sqrt(bestFit.loss / numPoints);
    const dataScale = (currentScenario === 's1' || currentScenario === 's3') ? (b - a) : (q3 - q1);
    
    if (dataScale > 0 && (rmse / dataScale) > 2) {
        warnings.push("Warning: The best-fit distribution still has a large error relative to the data range. The result may not be reliable.");
    } // 这里需要修改，指明我们的方法是数据驱动的方法
    
    return { bestFit, warnings };
}
