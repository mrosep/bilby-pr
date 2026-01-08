# bilby-pr

Posterior repartitioning in bilby using normalizing flows (margarine_unbounded).

This package extends bilby's nested sampling capabilities by using trained normalizing flows (MAFs) to repartition the prior during sampling, dramatically accelerating inference by focusing computational effort on high-probability regions of parameter space.

## What is Posterior Repartitioning?

Posterior repartitioning uses a trained normalizing flow learned from a preliminary guess at where the posterior lies (e.g., from a quick initial run or approximate posterior samples). During nested sampling, instead of sampling from the original Bayesian prior π(θ), we sample from the trained flow q(θ) and apply a repartitioning factor to the likelihood:

```
L_modified(θ) = L(θ) × π(θ) / q(θ)
```

This ensures that `L_modified(θ) × q(θ) = L(θ) × π(θ)`, so the product remains unchanged and we recover the original posterior from the true underlying Bayesian prior, but with far fewer likelihood evaluations concentrated where the posterior has mass.

## About margarine_unbounded

This package uses `margarine_unbounded`, a fork of the [`margarine` package by Bevins et al.](https://github.com/htjb/margarine) that is modified to remove implicit bounds on parameters when learning flows. This makes the flows "unbounded", allowing for cleaner transformations and better handling of parameters that may approach prior boundaries. The key differences are:

- Uses unbounded transformations (no manual clipping required)
- Provides `.quantile()` method for clean uniform→physical parameter transforms
- Handles standardization/unstandardization internally

## Key Features

- **Uses margarine_unbounded flows** with clean `.quantile()` and `.log_prob()` methods
- **Hybrid sampling**: Selected parameters use the flow, others use standard Bilby priors
- **Automatic reweighting**: Modified likelihood accounts for change of sampling prior
- **Multiprocessing support**: Works seamlessly with Bilby's parallel samplers
- **Simple API**: Just add two kwargs to your standard Bilby run

## Installation

### Standard Installation

```bash
# First install margarine_unbounded (if not already installed)
cd /path/to/margarine_unbounded
pip install .

# Then install bilby-pr
cd /path/to/bilby-pr
pip install .
```

### Development Installation

```bash
cd /path/to/bilby-pr
pip install -e .
```

## Usage

### Basic Example

```python
import bilby

# Set up your likelihood and priors as usual
likelihood = bilby.likelihood.GravitationalWaveTransient(...)
priors = bilby.prior.PriorDict(...)

# Specify which parameters are modeled by the flow
# IMPORTANT: Must be in the same order as used during training
flow_params = ['mass_ratio', 'chirp_mass', 'theta_jn', 'spin_1z', 'spin_2z']

# Run nested sampling with posterior repartitioning
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler='dynesty_pr',                      # Use the PR sampler
    weights_file='path/to/trained_flow.pkl',   # Path to trained MAF model
    flow_params=flow_params,                   # Parameters modeled by flow
    nlive=500,
    npool=4,                                    # Multiprocessing supported
    # ... other standard Dynesty kwargs
)
```

### Training a Flow

Before using posterior repartitioning, you need to train a normalizing flow on a preliminary guess at the posterior (e.g., samples from a quick initial run, approximate samples, or samples from a similar problem).

Example training workflow:
```python
from margarine_unbounded.maf import MAF

# Load samples from a preliminary run or approximate posterior
samples = ...  # Shape: (n_samples, n_parameters)

# Train the flow
maf = MAF(samples, number_networks=10, hidden_layers=[128, 128])
maf.train(epochs=100)

# Save the trained model
maf.save('trained_flow.pkl')
```

## How it Works

### 1. Prior Transform

For flow-modeled parameters:
```
Uniform [0,1] → Standard Normal N(0,1) → MAF → Physical Parameters
                    (via quantile function)
```

For other parameters:
```
Uniform [0,1] → Physical Parameters
    (via standard Bilby prior.rescale())
```

### 2. Modified Likelihood

To account for sampling from q(θ) instead of π(θ), the likelihood is modified:

```
L_modified(θ) = L(θ) × π(θ) / q_new(θ)
```

where:
- `L(θ)` is the original likelihood
- `π(θ)` is the original prior for all parameters
- `q_new(θ) = q(θ_flow) × π(θ_non-flow)` is the repartitioned prior
  - `q(θ_flow)`: flow density for flow-modeled parameters
  - `π(θ_non-flow)`: original prior for non-flow parameters

### 3. Nested Sampling

The modified likelihood is used with the repartitioned prior transform in Dynesty's nested sampling loop. Since `L_modified(θ) × q_new(θ) = L(θ) × π(θ)`, the final posterior samples correctly represent `p(θ|data)` from the original Bayesian prior.

## Requirements

- Python >= 3.9
- bilby >= 2.3.0
- margarine_unbounded
- tensorflow >= 2.8.0
- tensorflow-probability >= 0.16.0
- dynesty
- numpy

## Multiprocessing Notes

The sampler handles multiprocessing automatically. Each worker process:
1. Loads its own copy of the trained flow
2. Disables GPU to avoid memory conflicts (uses CPU only)
3. Configures TensorFlow threading for optimal CPU performance

## Troubleshooting

### Parameter Order

**Critical**: The `flow_params` list must be in the **same order** as used during flow training. Mismatched ordering will produce incorrect results.

### Flow Support

If you get `-inf` log-likelihoods, check that:
1. Your trained flow covers the region where the posterior has mass
2. The flow was trained on samples in the same physical parameter space (not rescaled)
3. The parameter names in `flow_params` match your training data

### TensorFlow Warnings

You may see TensorFlow warnings about CPU features or threading. These are usually harmless and can be ignored. The sampler explicitly configures TensorFlow for CPU-only operation to ensure stability with multiprocessing.

## Citation

If you use this package, please cite:
- The Bilby paper: [Ashton et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJS..241...27A)
- The margarine paper: [Bevins et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.2923B)
- Posterior repartitioning for GWs papers: [Prathaban et al. 2025](https://academic.oup.com/mnras/article/541/1/200/8163830) and [add arXiv link for simple-pe-PR]()

## License

MIT License - see LICENSE file for details
