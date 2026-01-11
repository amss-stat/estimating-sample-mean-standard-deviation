import { initializeModels, calculateBestDistribution } from './logic.js';

let currentScenario = 's1';

const ui = {
    button: document.getElementById('predict-button'),
    resetButton: document.getElementById('reset-button'),
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

const SCENARIO_LABELS = {
    s1: "S1 (Min, Median, Max)",
    s2: "S2 (Q1, Median, Q3)",
    s3: "S3 (Min, Q1, Median, Q3, Max)"
};

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
    ui.resultText.textContent = ''; 
    ui.status.textContent = 'Please enter new values for the selected scenario.';
}

async function init() {
    ui.status.textContent = '⏳ Loading all models, please wait...';
    ui.button.disabled = true;
    try {
        await initializeModels();
        ui.status.textContent = '✅ Models loaded. Please select a scenario and enter your data.';
        ui.button.disabled = false;
    } catch (error) {
        console.error(error);
        ui.status.textContent = '❌ Failed to load models. Please check your network connection and refresh.';
    }
}

function getAndValidateInputs() {
    const inputs = {};
    for (const key in ui.inputs) {
        inputs[key] = parseFloat(ui.inputs[key].value);
    }
    const { n: n_original, m, a, b, q1, q3 } = inputs;

    const requiredInputs = currentScenario === 's1' ? [n_original, m, a, b] :
                           currentScenario === 's2' ? [n_original, m, q1, q3] :
                           [n_original, m, a, b, q1, q3];
    
    if (requiredInputs.some(isNaN)) {
        alert("Please enter valid numbers for all fields.");
        return null;
    }
    
    if (n_original < 10 || !Number.isInteger(n_original)) {
        alert("Input validation failed: sample size must be an integer greater than 10.");
        return null;
    }

    const hasNegative = (currentScenario === 's1' || currentScenario === 's3') ? a < 0 : (currentScenario === 's2' ? q1 < 0 : false);
    if (hasNegative) {
        alert("Negative values detected: please check your data or apply a transformation. This tool currently only works for data with positive values.");
        return null;
    }

    const validationMap = { 
        s1: a <= m && m <= b, 
        s2: q1 <= m && m <= q3, 
        s3: a <= q1 && q1 <= m && m <= q3 && q3 <= b 
    };
    
    if (!validationMap[currentScenario]) {
        alert("Input validation failed: values are not in logical order (e.g., min <= median <= max).");
        return null;
    }
    return inputs;
}

function displayResults(bestFit, sourceScenario) {
    let output = `Input Scenario: ${SCENARIO_LABELS[sourceScenario]}\n`;
    output += `Best Fit Distribution: ${bestFit.name}\n`;
    output += `-------------------------------------------\n`;
    output += `Estimated Sample Mean : ${bestFit.mean.toFixed(4)}\n`;
    output += `Estimated Sample SD   : ${bestFit.std.toFixed(4)}\n\n`;
    output += `Best Fit Distribution Parameters:\n`;
    const paramsString = Object.entries(bestFit.params)
        .map(([key, value]) => `  ${key.padEnd(7)}: ${value.toFixed(4)}`)
        .join('\n');
    output += paramsString;
    
    ui.resultText.textContent = output;
    ui.resultContainer.classList.remove('hidden');
}

async function handlePredictionClick() {
    const inputs = getAndValidateInputs();
    if (!inputs) return;

    ui.button.disabled = true;
    ui.status.textContent = '🧠 Calculating...';
    ui.resultContainer.classList.add('hidden');
    
    try {
        const { bestFit, warnings } = await calculateBestDistribution(currentScenario, inputs);
        displayResults(bestFit, currentScenario);
        
        let statusMessage = '✅ Calculation complete.';
        if (warnings && warnings.length > 0) {
            statusMessage += ` ${warnings.join(' ')}`;
        }
        ui.status.textContent = statusMessage;

    } catch (error) {
        if (error.name === "InvalidInputError") {
            ui.status.textContent = `❌ Input Error: ${error.message}`;
        } else {
            console.error("An unexpected error occurred during prediction:", error);
            ui.status.textContent = '❌ An unexpected error occurred. Please check the console or contact the author.';
        }
    } finally {
        ui.button.disabled = false;
    }
}

// --- Reset Button ---
function handleResetClick() {
    // Clear all input fields
    for (const key in ui.inputs) {
        ui.inputs[key].value = '';
    }

    // Hide the result container and reset messages
    ui.resultContainer.classList.add('hidden');
    ui.resultText.textContent = '';
    ui.status.textContent = 'Please enter your data.';

    // Set focus back to the first input for convenience
    ui.inputs.n.focus();
}


// --- Event Binding ---
window.addEventListener('DOMContentLoaded', () => {
    init();
    updateUIForScenario('s1');
});
ui.button.addEventListener('click', handlePredictionClick);
ui.resetButton.addEventListener('click', handleResetClick); // Add listener for reset button
ui.scenarioRadios.forEach(radio => radio.addEventListener('change', e => updateUIForScenario(e.target.value)));
