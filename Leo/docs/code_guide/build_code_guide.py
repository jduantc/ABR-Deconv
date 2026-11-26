"""Build an offline, browsable guide to the source files in Leo."""

import ast
import hashlib
import html
import re
from pathlib import Path

LEO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

ANNOTATED_SHA256 = {
    'data.py': 'f85ca58e5d69f292a7bfe97dc77464fae98cbd86325c2065514f8130c16f2166',
    'diffusion.py': '8b26b97eb4b219851e34a33c25bc08b3f2b377df3386eb2ed023c61253ed491f',
    'model.py': '7b1d6fd9612064a08af136afa68d1ad8305ff54c89cb286136dd8c05fe61ef4a',
    'train.py': 'd5bf9f113d78bec52571a1973fa95c2cb446f43f22b9684d55b17ccdd81e21b8',
    'dps/reconstruct.py': 'cfb661c7342094060a737d100ec96ff4d6a63ada06c8e34495f377ddcd70a896',
    'dps/demo.py': '7fe2f6f7d55b9ad4d2bf7b9d6e074309b355e18aad02c39f2c72f038c070ca9e',
    'dps/run_eeg.py': '13e33c7805252d8208076a7f76557be35e54032032558ba6bfb26ac62246fcbc',
}

# These are file-specific explanations of the scientific core. Line numbers
# refer to the exact source snapshot embedded in the generated document.
DETAIL = {
    'data.py': {
        10: 'The waveforms use 10,000 samples per second.',
        11: 'Each ABR has 256 samples.',
        12: 'One model amplitude unit corresponds to 0.1 microvolts.',
        13: 'Build the time axis in milliseconds; sample 50 is the stimulus onset.',
        17: 'Generate a batch of artificial clean ABRs. The seed controls the random choices.',
        18: 'Create a reproducible random number generator.',
        19: 'Create a longer time axis so filtering can happen before the final crop.',
        21: 'Randomize the latency of the main peak for each waveform.',
        22: 'Randomize the width of the main peak.',
        23: 'Randomize and limit the main peak amplitude.',
        24: 'Choose how far the negative trough comes after the main peak.',
        25: 'Choose the trough size relative to the peak.',
        26: 'Choose the width of the trough.',
        27: 'Choose the size of an earlier positive component.',
        28: 'Choose when that earlier component occurs.',
        30: 'Define a helper for a bell-shaped waveform component.',
        31: 'Evaluate a Gaussian curve at every point of the long time axis.',
        33: 'Start each synthetic ABR with its main positive peak.',
        34: 'Subtract a later trough; the expression continues below.',
        37: 'Add a smaller earlier peak.',
        38: 'Set the unfiltered response before the stimulus to zero.',
        41: 'Create an 80–1500 Hz Butterworth bandpass filter.',
        42: 'Filter each waveform forwards and backwards.',
        43: 'Crop the long, filtered signal to the 256-sample ABR window.',
        45: 'Randomly select about 10% of examples to be null responses.',
        46: 'Replace the selected responses with zero waveforms.',
        47: 'Return the generated examples and the mask identifying null examples.',
        59: 'Generate the requested synthetic training waveforms.',
        61: 'Save the examples and their metadata as a compressed NPZ file.',
    },
    'diffusion.py': {
        8: 'The standard schedule contains 200 diffusion steps.',
        11: 'Create the preselected cosine schedule; this is not inferred from an EEG recording.',
        12: 'Create equally spaced points from the beginning to the end of the schedule.',
        13: 'Evaluate the cosine-squared curve describing retained signal.',
        14: 'Convert neighboring curve values into per-step forward noise variances and cap them for stability.',
        15: 'Multiply the per-step signal-retention factors to obtain cumulative alpha_bar values.',
        18: 'Create a noisy input and the known v target for training.',
        19: 'For each example, compute the square root of its cumulative signal coefficient.',
        20: 'Compute the corresponding coefficient for the generated diffusion noise.',
        21: 'Mix each clean ABR with generated noise at its sampled timestep.',
        22: 'Compute the v target from the same clean ABR and noise. The model learns to predict this.',
        23: 'Return both the noisy model input and its known target.',
        26: 'Turn the network’s v prediction and noisy waveform into a clean-waveform estimate.',
        27: 'Get the signal coefficient for the requested timestep.',
        28: 'Get the noise coefficient for the requested timestep.',
        29: 'Apply the v-parameterization identity. An imperfect v prediction gives an imperfect clean estimate.',
    },
    'model.py': {
        11: 'Define the fully connected residual MLP denoiser.',
        14: 'Accept 256 waveform values plus 64 timestep features and project them into 256 hidden values.',
        15: 'Create residual blocks; the following lines define their layers.',
        18: 'Normalize hidden activations before applying the block’s nonlinear transformations.',
        19: 'Apply the SiLU nonlinearity.',
        20: 'Apply a learned linear transformation to the hidden values.',
        22: 'Apply a second learned linear transformation.',
        24: 'Build three such residual blocks.',
        27: 'Create the output layers that return one v prediction per ABR point.',
        30: 'Store fixed frequencies used to encode the timestep; they are saved with the model.',
        34: 'Run the model on noisy waveforms and their diffusion-step numbers.',
        36: 'Multiply the step number by fixed frequencies to form angles.',
        37: 'Concatenate the waveform with 32 sine and 32 cosine timestep features.',
        38: 'Project those input features to the hidden representation.',
        39: 'Apply each residual block in sequence.',
        40: 'Add a scaled block output to the existing hidden representation.',
        41: 'Return 256 predicted v values per waveform.',
        44: 'Define the alternative dilated convolutional denoiser.',
        52: 'Convert the single input waveform into a set of learned feature channels.',
        53: 'Project the timestep features into the same channel space.',
        54: 'Create convolutional residual blocks with multiple dilation sizes.',
        63: 'Increase the spacing of convolution taps to capture patterns across different time scales.',
        66: 'Convert the feature channels back to a single predicted waveform.',
        69: 'Build the timestep angles for the CNN.',
        70: 'Transform sine/cosine timestep features into a per-channel conditioning signal.',
        71: 'Add the timestep conditioning to the waveform features.',
        72: 'Run the CNN residual blocks.',
        73: 'Add each scaled residual-block output to the current features.',
        74: 'Return the single-channel v prediction.',
        77: 'Define a timestep-conditioned residual block used by the 1D U-Net.',
        91: 'Produce a learned scale and shift from the timestep representation.',
        92: 'Use the scale and shift to condition the normalized features on time.',
        94: 'Add the skip connection and residual branch, with a scaling factor.',
        97: 'Define the 1D U-Net architecture used in architecture comparisons.',
        123: 'Run the encoder stages and save their features for skip connections.',
        125: 'Downsample the waveform features by a factor of two.',
        127: 'Remember this resolution for a later skip connection.',
        129: 'Run the decoder while matching earlier encoder features.',
        130: 'Upsample features to match the selected skip resolution.',
        131: 'Concatenate the skip features and apply the conditioned decoder block.',
        132: 'Return the v prediction with one value per waveform point.',
        135: 'Choose the network architecture by name; the default is the MLP.',
        138: 'Construct and return the selected architecture.',
    },
    'train.py': {
        17: 'Train a denoiser using the settings supplied through args.',
        18: 'Select CPU or CUDA according to the requested device.',
        20: 'Set the PyTorch random seed for this training run.',
        22: 'Open the previously generated training dataset.',
        23: 'Load the clean ABRs as 32-bit floating-point values.',
        24: 'Read the data-generation seed for the saved run settings.',
        25: 'Check that the dataset is a collection of 256-point waveforms.',
        27: 'Reject an empty dataset or values that are not finite.',
        30: 'Choose a folder in which to record this run.',
        32: 'Do not accidentally overwrite an existing model or loss log.',
        37: 'Start a dictionary of run settings and data provenance.',
        56: 'Write those settings to settings.json.',
        60: 'Copy the clean training data to the chosen device.',
        61: 'Create the requested MLP, CNN, or U-Net denoiser.',
        62: 'Create a separate EMA copy: a smoothed history of model weights for evaluation.',
        63: 'Create the AdamW optimizer, which updates model weights during training.',
        64: 'Build the diffusion noise schedule on the same device.',
        67: 'Generate separate synthetic validation waveforms with a different seed.',
        69: 'Create a reproducible generator for the fixed validation noise and timesteps.',
        70: 'Sample the validation diffusion step for each validation waveform.',
        73: 'Generate validation diffusion noise once, then reuse it for fair checkpoint comparisons.',
        74: 'Prepare the fixed noisy validation inputs and known v targets.',
        78: 'Initialize the best validation loss to infinity.',
        79: 'Start a list of recent training losses to average when logging.',
        84: 'Open the CSV file that will store training and validation losses.',
        88: 'Write the column names to the CSV.',
        89: 'Perform the requested number of optimizer updates. This step is not a diffusion timestep.',
        90: 'Randomly select a batch of clean examples, with replacement.',
        93: 'Collect the selected clean waveforms.',
        94: 'Choose a random diffusion timestep independently for each batch example.',
        95: 'Generate Gaussian diffusion noise shaped like the clean batch.',
        96: 'Mix clean waveforms and noise and compute their known v targets.',
        98: 'Ask the neural network to predict v from the noisy waveforms and their timesteps.',
        99: 'Compute mean squared error between predicted v and target v. This trains denoising, not ABR reconstruction RMSE.',
        100: 'Clear gradients left by the previous optimizer update.',
        101: 'Backpropagate the training loss to compute gradients of the model weights.',
        102: 'Limit the total gradient norm to 1 to control large updates.',
        103: 'Update the neural network weights using AdamW.',
        106: 'Update EMA weights without tracking gradients.',
        107: 'Pair each EMA parameter with the corresponding trained model parameter.',
        110: 'Move the EMA parameter 0.5% toward the current trained parameter.',
        111: 'Record the numeric training loss for the next log entry.',
        113: 'At requested training-update counts, save an EMA checkpoint.',
        124: 'At logging intervals and the final step, evaluate and save a loss row.',
        125: 'Validation does not update weights or need weight gradients.',
        126: 'Predict v for the fixed validation examples using EMA weights.',
        127: 'Calculate validation v-prediction mean squared error.',
        130: 'Create a row containing the update number and training and validation losses.',
        135: 'Append the row to the CSV file.',
        136: 'Flush buffered text so progress is visible on disk.',
        142: 'Clear the recent-loss list for the next logging interval.',
        143: 'If validation v-MSE is lower than before, keep this EMA model as the best.',
        145: 'Save the best EMA weights and their metadata as model.pt.',
        160: 'Read command-line options, validate them, and start training.',
        165: 'Default to 8,000 optimizer updates; this is independent of the 200 diffusion steps.',
        166: 'Default to 256 randomly selected training examples per optimizer update.',
        169: 'Choose the network architecture; the default is mlp.',
        189: 'Start the training routine after checking the arguments.',
    },
    'dps/reconstruct.py': {
        15: 'Define a matrix that converts one candidate ABR into the EEG predicted from the pulse timings.',
        17: 'Convert pulse weights to a 32-bit tensor.',
        18: 'Check that pulses is a nonempty one-dimensional sequence.',
        19: 'Stop with a useful error if pulse data are invalid.',
        21: 'Make a column of EEG sample indices, one row for each recorded sample.',
        22: 'Make a row of ABR offsets; sample 50 is aligned to each pulse onset.',
        23: 'For each EEG sample and ABR point, calculate which pulse index contributes.',
        24: 'Mark only pulse indices that fall inside the recording.',
        25: 'Look up pulse weights and zero out invalid indices. The result is an EEG-length by 256 matrix.',
        28: 'Load a previously trained model checkpoint for reconstruction.',
        29: 'Read the checkpoint onto the CPU using the weights-only loader.',
        30: 'Create the recorded architecture; old checkpoints default to the MLP.',
        31: 'Load the saved trained weights into the model.',
        32: 'Switch the network to evaluation mode.',
        33: 'Visit each network parameter.',
        34: 'Freeze its weights. Reconstruction needs gradients with respect to current, not weight updates.',
        35: 'Return the ready-to-use network.',
        38: 'Reconstruct one ABR from a model, observed EEG, and pulse train. Guidance and seed are keyword arguments.',
        44: 'Require a nonnegative guidance strength.',
        46: 'Convert the observed EEG to a 32-bit tensor in model amplitude units.',
        47: 'Build the forward matrix that maps a candidate ABR to predicted EEG.',
        48: 'Check that the EEG is one-dimensional and has the same length as the pulse train.',
        50: 'Build the same kind of 200-step noise schedule used by this model.',
        51: 'Create a seeded generator for repeatable random sampling.',
        52: 'Start with a random 256-point noisy waveform; it is not an EEG recording.',
        54: 'Loop backward through the 200 diffusion timesteps, from 199 to 0.',
        55: 'Forget the previous step’s computation history and enable a fresh gradient with respect to current.',
        56: 'Package t as an integer tensor so the network knows the current noise level.',
        57: 'Use the trained network to predict v for this noisy waveform and timestep.',
        58: 'Combine current and predicted v to estimate the clean ABR.',
        59: 'Apply the pulse matrix to the estimated clean ABR to predict the recording.',
        62: 'Subtract predicted EEG from observed EEG, sample by sample.',
        63: 'Compute the RMS EEG residual. This standalone loss does not divide by noise standard deviation.',
        64: 'Differentiate the EEG loss through the clean estimate and model with respect to the current noisy waveform.',
        67: 'Read the cumulative diffusion-schedule value at the current timestep.',
        68: 'Read the value for the next cleaner step; use 1 at the clean endpoint.',
        69: 'Convert the neighboring cumulative values into the per-step signal-retention factor.',
        70: 'Compute the corresponding forward-step diffusion-noise variance.',
        71: 'Start the reverse-diffusion mean with the schedule-weighted clean estimate.',
        72: 'Add the schedule-weighted current noisy waveform to that mean.',
        73: 'Calculate the variance for random sampling of the next diffusion state.',
        74: 'Draw seeded Gaussian diffusion noise, except at the final step where it is zero.',
        75: 'Combine the diffusion mean, scheduled randomness, and the negative EEG-gradient correction.',
        77: 'Detach the final result, convert it to a NumPy array, and remove the batch dimension.',
    },
    'dps/demo.py': {
        24: 'Generate artificial ABRs for this self-contained demonstration.',
        25: 'Select a nonzero synthetic waveform as the known reference.',
        26: 'Create a seeded generator for artificial measurement noise.',
        27: 'Create a half-second pulse train filled with zeros.',
        28: 'Place a sequence of unit pulses in that train.',
        29: 'Choose the standard deviation of the simulated EEG noise. This generates fake EEG; it is not an input to the standalone guidance loss.',
        30: 'Construct noise-free synthetic EEG by applying the pulse matrix to the known ABR.',
        31: 'Add generated measurement noise to form the synthetic observed EEG.',
        34: 'Load the trained denoiser checkpoint.',
        35: 'Reconstruct an ABR from the simulated EEG and pulses.',
        36: 'Compare the reconstruction with the known synthetic reference and convert RMSE to microvolts.',
        38: 'Save the example inputs, reference, estimate and settings to an NPZ file.',
        49: 'Print reconstruction RMSE in microvolts.',
    },
    'dps/run_eeg.py': {
        15: 'Define the command-line entry point for a prepared EEG recording.',
        17: 'Require a trained model checkpoint path.',
        18: 'Require an NPZ input containing eeg_uv and pulses arrays.',
        19: 'Accept the EEG-guidance strength.',
        20: 'Accept a path for the reconstructed output.',
        25: 'Open the prepared EEG and pulse-train file.',
        26: 'Convert EEG from microvolts to model units (one unit is 0.1 microvolts).',
        27: 'Read pulse-event weights; these are not EEG amplitudes.',
        28: 'Load and freeze the trained model.',
        29: 'Call the standalone reconstruction using EEG, pulses, and the chosen guidance.',
        35: 'Create the output directory if needed.',
        36: 'Save the estimated ABR, time axis and guidance setting.',
        38: 'Convert the estimated ABR back from model units to microvolts.',
    },
}

DESCRIPTIONS = {
 'data.py': 'Generates synthetic clean training ABRs and defines their units and time axis.',
 'diffusion.py': 'Defines the 200-step noise schedule, training targets, and conversion from predicted v to a clean estimate.',
 'model.py': 'Defines the MLP, CNN and U-Net architectures that predict v from a noisy waveform and timestep.',
 'train.py': 'Samples noisy training examples, trains the denoiser, evaluates v-MSE and saves EMA checkpoints.',
 'dps/reconstruct.py': 'The main standalone DPS reconstruction: pulse matrix, model loading, EEG guidance and reverse diffusion.',
 'dps/demo.py': 'Creates synthetic EEG from a known ABR, reconstructs it and reports reference ABR RMSE.',
 'dps/run_eeg.py': 'Loads a prepared real or simulated EEG NPZ, applies the standalone reconstruction and saves an ABR.',
 'experiments/datasets.py': 'Downloads and prepares EARNDB-derived subject-separated datasets and synthetic training pipelines.',
 'experiments/study.py': 'Runs the architecture, training-duration and guidance comparison, including synthetic observations and evaluation.',
 'experiments/reconstruction_suite.py': 'Runs reconstruction-focused training, checkpoint and guidance selection, and ridge comparisons.',
 'experiments/reconstruction_report.py': 'Turns reconstruction-focused results into figures and an HTML report; can publish the report.',
 'experiments/plots.py': 'Creates presentation figures for the original comparison study.',
 'experiments/fast_conv.py': 'Provides an equivalent small-channel convolution implementation used by Phoom comparison experiments.',
 'experiments/phoom_improvements.py': 'Runs the matched-budget Phoom-style MLP/CNN pilot, guidance selection and optional measurement refinement.',
 'experiments/phoom_matched_training.py': 'Continues a controlled Phoom-style MLP training trajectory toward 10 million optimizer updates.',
 'experiments/phoom_improvements_report.py': 'Builds and publishes the matched-budget pilot report and figures.',
 'experiments/phoom_matched_report.py': 'Creates interim and final reports for the long matched-training protocol.',
 'experiments/phoom_long_report.py': 'Creates the separate longer MLP convergence report.',
 'experiments/phoom_visual_comparison.py': 'Builds presentation figures comparing the Phoom-style experiments.',
 'experiments/wandb_monitor.py': 'Reads saved experiment logs and forwards progress to the W&B-compatible Forge dashboard.',
 'experiments/check_gpu.py': 'Checks that CUDA can perform the operations needed by training and reconstruction.',
 'experiments/test_study.py': 'Tests data splits, pulse timing, diffusion math, gradients and reproducibility.',
 'experiments/test_reconstruction.py': 'Tests the historical batched sampler and ensures truth is only used for diagnostics.',
 'experiments/test_phoom_improvements.py': 'Checks Phoom sampler equivalence, refinement and the accelerated convolution.',
 'experiments/test_phoom_matched_training.py': 'Checks that resuming long training preserves the next optimizer update and random stream.',
 'docs/reconstruction_guide/build_guide.py': 'Builds the separate HTML reconstruction tutorial and equation PNG files.',
 'experiments/install_nvidia_driver.sh': 'System setup script for an NVIDIA driver on the intended Debian machine; it modifies system packages.',
 'experiments/setup_gpu_env.sh': 'Creates a Python environment and installs the specified CUDA-enabled PyTorch and experiment dependencies.',
 'experiments/run_gpu.sh': 'Checks CUDA and runs the comparison study on the GPU.',
 'experiments/run_initial_comparison.sh': 'Runs the initial comparison up to 16,000 updates and records job status.',
 'experiments/run_server_job.sh': 'Runs the logged server GPU study after environment checks and tests.',
 'experiments/start_wandb_monitor.sh': 'Waits for existing Forge/W&B credentials and starts the monitoring script.',
 'requirements.txt': 'Minimum Python packages for the basic data, training and reconstruction code.',
 'experiments/requirements.txt': 'Additional packages used for datasets, figures, tests and Forge logging.',
 'pyproject.toml': 'Ruff lint settings for this folder.',
 '.gitignore': 'Rules describing local files that Git should ignore.',
 '.gitattributes': 'Git attributes for tracked files.',
}


def display(node):
    try:
        s = ast.unparse(node)
    except (ValueError, TypeError):
        return 'the expression'
    return s if len(s) <= 110 else s[:107] + '…'


def statement_hint(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        doc = ast.get_docstring(node)
        return f'Define the function {node.name}.' + (f' Its own description: {doc.splitlines()[0]}' if doc else ' The following indented lines are its body.')
    if isinstance(node, ast.ClassDef):
        doc = ast.get_docstring(node)
        return f'Define the class {node.name}.' + (f' {doc.splitlines()[0]}' if doc else ' Its indented body defines its behavior.')
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return 'Import the named module or objects so this file can use them.'
    if isinstance(node, ast.Assign):
        return f'Store the result of {display(node.value)} in {", ".join(display(x) for x in node.targets)}.'
    if isinstance(node, ast.AnnAssign):
        return f'Define {display(node.target)} with a type annotation' + (f' and assign {display(node.value)}.' if node.value else '.')
    if isinstance(node, ast.AugAssign):
        return f'Update {display(node.target)} using the expression on the right.'
    if isinstance(node, ast.Return):
        return 'Return ' + (display(node.value) if node.value else 'to the caller') + ' to the caller.'
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return f'For each value from {display(node.iter)}, assign it to {display(node.target)} and run the indented body.'
    if isinstance(node, ast.While):
        return f'Repeat the indented body while {display(node.test)} is true.'
    if isinstance(node, ast.If):
        if display(node.test) == "__name__ == '__main__'":
            return 'Run the indented entry point only when this file is executed directly.'
        return f'Run the indented branch only when {display(node.test)} is true.'
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return 'Enter a managed context for ' + ', '.join(display(x.context_expr) for x in node.items) + '; leave it when the indented block ends.'
    if isinstance(node, ast.Raise):
        return 'Stop this path by raising an error' + (f': {display(node.exc)}.' if node.exc else '.')
    if isinstance(node, ast.Assert):
        return f'Check that {display(node.test)} is true; fail the test or program if it is not.'
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return 'Documentation text describing this file, function, or class.'
        if isinstance(node.value, ast.Call):
            return f'Call {display(node.value.func)} with the arguments shown.'
        return 'Evaluate the expression shown.'
    if isinstance(node, ast.Try):
        return 'Start an error-handling block; later handlers describe what to do if an exception occurs.'
    if isinstance(node, ast.Break):
        return 'Leave the nearest enclosing loop.'
    if isinstance(node, ast.Continue):
        return 'Skip to the next iteration of the nearest loop.'
    if isinstance(node, ast.Pass):
        return 'Do nothing here; this statement is a placeholder.'
    return ''


def hints(path, source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.stmt) and hasattr(n, 'lineno'):
            text = statement_hint(n)
            if text:
                out[n.lineno] = text
    return out


def fallback(s, relative):
    line = s.strip()
    if line.startswith('#'):
        return 'Code author’s comment: ' + line.lstrip('#').strip()
    if relative.endswith('requirements.txt'):
        return 'Declare this package and any version constraints for the Python environment.'
    if relative == '.gitignore':
        return 'Ignore files or directories matching this pattern when Git looks for untracked files.'
    if relative == '.gitattributes':
        return 'Apply the shown Git attributes to paths matching this pattern.'
    if relative.endswith('.sh'):
        if line.startswith('set '):
            return 'Set shell execution options; these affect how the script handles errors and variables.'
        if line.startswith('export '):
            return 'Set an environment variable for commands launched by this script.'
        if line.startswith(('if ', 'elif ', 'while ', 'for ')):
            return 'Start or continue the shell condition or loop shown; its body follows.'
        if line in ('fi', 'done', 'then', 'do'):
            return 'Mark the boundary of the shell condition or loop above.'
        if line.startswith('cd '):
            return 'Change the working directory for the following shell commands.'
        if line.startswith(('exec ', 'bash ', 'sudo ', 'apt-get ', 'mkdir ', 'cp ', 'sha256sum ', 'date ', 'hostname', 'printf ', 'echo ')):
            return 'Run the shell command shown with its specified arguments.'
        if re.match(r'[A-Za-z_][A-Za-z_0-9]*=', line):
            return 'Assign the shown value to a shell variable.'
        if '.venv/bin/python' in line:
            return 'Run Python from the project environment with the script or arguments shown.'
    if line.startswith(('"""', "'''")):
        return 'Documentation string; this text describes the surrounding file or function.'
    if line in (')', ']', '}', '),', '],', '},', '):', ']:', '}:'):
        return 'Closes the expression or block started above.'
    if line.startswith(('else:', 'elif ', 'except ', 'finally:')):
        return 'Begin the alternative or error-handling branch shown.'
    if re.match(r'["\'][^"\']+["\']\s*:', line):
        return 'Add this named entry to the dictionary being built above.'
    if re.match(r'[A-Za-z_][A-Za-z_0-9]*\s*=', line):
        return 'Pass or define the named value as part of the expression above.'
    if line.startswith(('import ', 'from ')):
        return 'Import these objects for use by this file.'
    return 'Continues the expression, arguments, documentation, or command begun above; read it together with the surrounding lines.'


def files():
    out = [p for p in LEO.glob('*.py')]
    out += list((LEO / 'dps').glob('*.py'))
    out += list((LEO / 'experiments').glob('*.py'))
    out += list((LEO / 'experiments').glob('*.sh'))
    out += [LEO / 'docs/reconstruction_guide/build_guide.py']
    out += [LEO / x for x in ('requirements.txt', 'experiments/requirements.txt', 'pyproject.toml', '.gitignore', '.gitattributes')]
    order = ['data.py', 'diffusion.py', 'model.py', 'train.py', 'dps/reconstruct.py', 'dps/demo.py', 'dps/run_eeg.py']
    return sorted(out, key=lambda p: (0, order.index(p.relative_to(LEO).as_posix())) if p.relative_to(LEO).as_posix() in order else (1, p.relative_to(LEO).as_posix()))


def main():
    sections, nav = [], []
    for p in files():
        relative = p.relative_to(LEO).as_posix()
        source = p.read_text(encoding='utf-8')
        if relative in ANNOTATED_SHA256:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest != ANNOTATED_SHA256[relative]:
                raise ValueError(f'Review line notes and update the saved hash for {relative}')
        automated = hints(p, source) if p.suffix == '.py' else {}
        manual = DETAIL.get(relative, {})
        anchor = 'file-' + re.sub(r'[^a-z0-9]+', '-', relative.lower()).strip('-')
        description = DESCRIPTIONS.get(relative, 'Supporting project file.')
        nav.append(f'<a href="#{anchor}">{html.escape(relative)}</a>')
        rows = []
        for number, line in enumerate(source.splitlines(), 1):
            if not line.strip():
                continue
            explanation = manual.get(number) or automated.get(number) or fallback(line, relative)
            rows.append(f'<tr id="{anchor}-L{number}"><td class="line">{number}</td><td class="source"><code>{html.escape(line)}</code></td><td class="explain">{html.escape(explanation)}</td></tr>')
        detail_type = 'In-depth annotations' if relative in DETAIL else 'Brief reading hints'
        sections.append(f'<details class="file" id="{anchor}" data-search="{html.escape(relative.lower())}"><summary><span class="file-name">{html.escape(relative)}</span><span class="summary-text">{html.escape(description)}</span><span class="badge">{detail_type}</span></summary><div class="filebody"><p>{html.escape(description)}</p><div class="tablewrap"><table class="code-table"><thead><tr><th>Line</th><th>Your code</th><th>What this line does</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></div></details>')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Leo’s ABR code, explained</title><style>
:root{--ink:#172b40;--muted:#52667a;--accent:#176a91;--line:#dbe5ec;--bg:#f3f7fa}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,-apple-system,sans-serif}main{max-width:1450px;margin:auto;padding:46px 24px 80px}header{max-width:920px}h1{font-size:clamp(2.3rem,5vw,3.7rem);line-height:1.08;letter-spacing:-.035em;margin:12px 0 20px}h2{line-height:1.25;font-size:1.55rem;margin:0 0 14px}h3{margin:24px 0 7px}p{margin:10px 0 16px}a{color:#08638f}code{font:0.9em ui-monospace,SFMono-Regular,Consolas,monospace;background:#edf3f6;padding:2px 4px;border-radius:3px}.eyebrow{text-transform:uppercase;color:var(--accent);font-size:.77rem;letter-spacing:.14em;font-weight:750}.lead{font-size:1.12rem;color:var(--muted)}section,.file{background:white;border:1px solid var(--line);border-radius:13px;padding:26px 30px;margin:18px 0;box-shadow:0 4px 18px #243f530a}.callout{padding:15px 18px;background:#eaf5f9;border-left:4px solid #2382aa;border-radius:4px}.flow{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.flow span{flex:1;min-width:125px;padding:13px;background:#e9f3f7;border-radius:8px;text-align:center;font-size:.88rem;font-weight:700}.facts{border-collapse:collapse;width:100%}.facts th,.facts td{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:top}.facts th{color:#40596c}.nav{display:flex;flex-wrap:wrap;gap:7px}.nav a{background:#edf5f8;border-radius:5px;padding:4px 8px;font:0.79rem ui-monospace,monospace;text-decoration:none}.file{padding:0;scroll-margin-top:16px}.file summary{display:flex;gap:14px;align-items:center;cursor:pointer;padding:18px 22px;list-style:none}.file summary:before{content:'+';font-size:1.5rem;color:var(--accent);font-weight:600}.file[open] summary:before{content:'−'}.file-name{font:700 .95rem ui-monospace,monospace;min-width:210px}.summary-text{flex:1;color:var(--muted);font-size:.9rem}.badge{font-size:.72rem;background:#eaf4f8;color:#176080;border-radius:30px;padding:3px 9px;white-space:nowrap}.filebody{padding:0 20px 22px}.tablewrap{overflow:auto;max-height:75vh;border:1px solid var(--line);border-radius:8px}.code-table{border-collapse:collapse;width:100%;font-size:.86rem}.code-table thead{position:sticky;top:0;background:#eaf3f7;z-index:1}.code-table th{text-align:left;padding:9px;border-bottom:1px solid var(--line)}.code-table td{border-bottom:1px solid #edf2f5;vertical-align:top;padding:5px 9px}.code-table tr:nth-child(even){background:#fbfdfe}.code-table .line{color:#718496;text-align:right;width:45px;font:0.8rem ui-monospace,monospace}.code-table .source{min-width:360px;width:47%;background:#142a40;color:#e9f3f8}.code-table .source code{background:none;color:inherit;white-space:pre-wrap;overflow-wrap:anywhere;font-size:.81rem;padding:0}.code-table .explain{min-width:300px}input{font:inherit;width:min(100%,420px);padding:10px 12px;border:1px solid #adc4d0;border-radius:7px}@media(max-width:700px){main{padding:30px 12px}section{padding:21px 17px}.file summary{flex-wrap:wrap}.file-name{min-width:0}.summary-text{flex-basis:100%;order:3}.badge{margin-left:auto}.code-table .source{min-width:280px}.code-table .explain{min-width:240px}}
</style></head><body><main><header><div class="eyebrow">CS163 · A guide to your Leo folder</div><h1>Your ABR code, explained</h1><p class="lead">A map of the project and a line-by-line reading companion to the code you have on disk. Start with the flow, then open a file to see its source beside an explanation.</p></header>
<section><h2>1 · What are you building?</h2><p>Your goal is to estimate an auditory brainstem response (ABR) waveform from an EEG recording and the times of the sound pulses. First, a denoising network learns what training ABRs look like. Later, reconstruction combines that learned model with the observed EEG.</p><div class="flow"><span>Clean training ABRs</span><span>Add diffusion noise</span><span>Train network to predict v</span><span>Save trained weights</span></div><div class="flow"><span>EEG + pulse times</span><span>Start from noise</span><span>Estimate a clean ABR</span><span>Use EEG to guide it</span><span>Repeat → estimated ABR</span></div><p class="callout"><strong>Training and reconstruction are different.</strong> During training, the optimizer changes the network weights to improve v prediction. During standalone reconstruction, the loaded weights are frozen; the waveform is adjusted using the network and EEG.</p></section>
<section><h2>2 · Follow one example through the files</h2><ol><li><strong>Make examples.</strong> <code>data.py</code> creates 256-point synthetic clean ABRs. One model amplitude unit represents 0.1 µV.</li><li><strong>Define the noise process.</strong> <code>diffusion.py</code> creates the 200-step schedule and turns each clean waveform into a noisy input and known v target.</li><li><strong>Build the network.</strong> <code>model.py</code> defines MLP, CNN and U-Net options. Each predicts v from a noisy waveform and a timestep.</li><li><strong>Train it.</strong> <code>train.py</code> compares predicted v with target v, updates weights and saves an EMA checkpoint. Its validation loss is v-prediction MSE.</li><li><strong>Reconstruct.</strong> <code>dps/reconstruct.py</code> loads the model, starts with random noise, repeatedly estimates a clean ABR, compares its predicted EEG to the observed EEG, and uses that gradient to guide the next sample.</li><li><strong>Evaluate.</strong> <code>dps/demo.py</code> generates synthetic EEG from a known ABR, reconstructs it and computes ABR RMSE against that reference. <code>dps/run_eeg.py</code> instead accepts a prepared EEG and pulse-train file.</li></ol></section>
<section><h2>3 · Three measurements that answer different questions</h2><table class="facts"><tr><th>Measurement</th><th>Comparison</th><th>Why it exists</th></tr><tr><td>Training / validation v-MSE</td><td>Predicted v versus the known v training target.</td><td>Teaches and evaluates the denoising network; it is not reconstruction RMSE.</td></tr><tr><td>EEG guidance loss</td><td>Observed EEG versus the EEG predicted from the current clean-ABR estimate.</td><td>Produces a gradient used to steer standalone reconstruction. It is RMS across EEG samples.</td></tr><tr><td>Reconstruction RMSE</td><td>The final reconstructed ABR versus a known reference ABR.</td><td>Measures waveform recovery when a reference exists, for example in a simulation.</td></tr></table><p>The current standalone guidance loss does not divide by noise standard deviation. Some historical batched experiments and the separate Phoom code retain their original normalization, so their guidance values and reported results need that context. The noise added in the synthetic demo is used to <em>create</em> simulated EEG; it is not supplied to the standalone loss.</p></section>
<section><h2>4 · What are all the folders?</h2><table class="facts"><tr><th>Folder / file</th><th>What it contains</th></tr><tr><td><code>dps/</code></td><td>Your standalone reconstruction function and two ways to run it.</td></tr><tr><td><code>experiments/</code></td><td>Dataset preparation, experiment runners, Phoom comparisons, reporting, tests and server helpers. Some runners use separate implementations and protocols.</td></tr><tr><td><code>results/</code></td><td>Saved reports, figures, tables and selected prediction arrays. These are outputs, not programs that train the network.</td></tr><tr><td><code>runs/</code></td><td>Local run data such as checkpoints, logs, training/validation tables and working outputs. Contents differ by run and some large files are not in Git.</td></tr><tr><td><code>docs/reconstruction_guide/</code></td><td>A focused HTML tutorial and downloadable equation images for the standalone reconstruction.</td></tr><tr><td><code>README.md</code> and <code>experiments/*.md</code></td><td>Written run instructions, methods and experiment notes. Read these as documentation rather than executable source.</td></tr><tr><td><code>__pycache__/</code>, <code>.pytest_cache/</code>, <code>.ruff_cache/</code></td><td>Temporary Python, test and lint caches.</td></tr><tr><td><code>docs/code_guide/</code></td><td>This guide and its builder; the builder is a documentation generator, not part of training or reconstruction.</td></tr></table><p>Common output formats: <code>.pt</code> stores model or training state; <code>.npz</code> stores arrays; <code>.csv</code> stores tables; <code>.json</code> stores settings or metadata; <code>.png</code>, <code>.pdf</code> and <code>.svg</code> are figures. The <code>phoom/</code> implementation lives outside <code>Leo/</code>; some experiment scripts import it.</p><p>Useful reports: <a href="../../results/reconstruction_focus/REPORT.html">reconstruction-focused report</a> · <a href="../../results/phoom_improvements/REPORT.html">Phoom pilot report</a> · <a href="../../results/phoom_visual_comparison/COMPARISON.html">visual comparison</a> · <a href="../reconstruction_guide/RECONSTRUCTION_GUIDE.html">detailed reconstruction tutorial</a>.</p></section>
<section><h2>5 · How to use the code reader</h2><p>Select a file below to expand it. Each row pairs a line from the actual source snapshot with a reading note. Blank lines are omitted; original line numbers are kept. The seven core Python files have detailed scientific explanations. The remaining experiment, test, shell and configuration files have shorter reading hints: multi-line expressions must be read together, and an automatically generated hint is not a full explanation of every scientific choice.</p><p><strong>Suggested order:</strong> <code>data.py</code> → <code>diffusion.py</code> → <code>model.py</code> → <code>train.py</code> → <code>dps/reconstruct.py</code> → <code>dps/demo.py</code> → <code>dps/run_eeg.py</code>.</p><p>Search filters by filename. Select a filename below, or scroll down and expand a file. The source is a snapshot taken when this guide was built. If a core file changes, the builder requires its detailed line notes to be reviewed before it can be rebuilt.</p><label for="filter">Filter files</label><br><input id="filter" placeholder="e.g. reconstruct or phoom"><div class="nav">__NAV__</div></section>
__SECTIONS__
<section><h2>A note about the experiments</h2><p>Several experiment runners deliberately preserve the implementation used when their results were recorded. In particular, a batched experiment’s noise scaling, data generation, model architecture, or selection procedure may differ from the standalone reconstruction. A simulated reference is known because it was used to construct the synthetic EEG; real EEG generally does not provide that exact ground truth.</p><p>This is an offline HTML document with the code embedded. The companion builder is <code>docs/code_guide/build_code_guide.py</code>; it creates this reading guide and is not needed to train or reconstruct.</p></section></main><script>
const filter=document.querySelector('#filter');filter.addEventListener('input',()=>{const q=filter.value.toLowerCase().trim();document.querySelectorAll('.file').forEach(e=>e.hidden=!e.dataset.search.includes(q));document.querySelectorAll('.nav a').forEach(a=>a.hidden=!a.textContent.toLowerCase().includes(q));});document.querySelectorAll('.nav a').forEach(a=>a.addEventListener('click',()=>{const e=document.querySelector(a.getAttribute('href'));if(e)e.open=true;}));
</script></body></html>'''
    HERE.mkdir(parents=True, exist_ok=True)
    target = HERE / 'LEO_CODE_GUIDE.html'
    target.write_text(page.replace('__NAV__', ''.join(nav)).replace('__SECTIONS__', ''.join(sections)), encoding='utf-8')
    print(f'Wrote {target} with {len(sections)} source files.')


if __name__ == '__main__':
    main()
