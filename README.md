# bilby-pr

Posterior repartitioning in bilby using normalizing flows (margarine or margarine_unbounded).

This package extends bilby's nested sampling capabilities by using trained normalizing flows (MAFs) to repartition the prior during sampling, dramatically accelerating inference by focusing computational effort on high-probability regions of parameter space.

## What is Posterior Repartitioning?

Posterior repartitioning uses a trained normalizing flow learned from a preliminary guess at where the posterior lies (e.g., from a quick initial run or approximate posterior samples). During nested sampling, instead of sampling from the original Bayesian prior π(θ), we sample from the trained flow q(θ) and apply a repartitioning factor to the likelihood:

```
L_modified(θ) = L(θ) × π(θ) / q(θ)
```

This ensures that `L_modified(θ) × q(θ) = L(θ) × π(θ)`, so the product remains unchanged and we recover the original posterior from the true underlying Bayesian prior, but with far fewer likelihood evaluations concentrated where the posterior has mass.

## Flow Backends

The flow can be trained with either of two packages, chosen with the `flow_backend` kwarg:

| `flow_backend` | Package | Support |
|---|---|---|
| `"margarine"` | [`margarine`](https://github.com/htjb/margarine) by Bevins et al., TensorFlow versions (>=1.2.8, <2) | Bounded to the box `[theta_min, theta_max]` set when training |
| `"margarine_unbounded"` | [`margarine_unbounded`](https://github.com/mrosep/margarine_unbounded), a fork of margarine | Unbounded |

`margarine_unbounded` is modified to remove the implicit bounds on parameters when learning flows. The key differences are:

- Uses unbounded transformations (no manual clipping required)
- Provides `.quantile()` method for clean uniform→physical parameter transforms
- Handles standardization/unstandardization internally (mean/std instead of min/max)

`flow_backend` must match the package the flow was trained with; there is no default. Loading a flow with the wrong backend does not necessarily raise an error and can give wrong densities.

margarine 2.x (rewritten in JAX) is not supported yet.

## Key Features

- **Uses margarine or margarine_unbounded flows** through a common `.quantile()` / `.log_prob()` adapter (`bilby_pr/flows.py`)
- **Hybrid sampling**: Selected parameters use the flow, others use standard Bilby priors
- **Automatic reweighting**: Modified likelihood accounts for change of sampling prior
- **Multiprocessing support**: Works seamlessly with Bilby's parallel samplers
- **Simple API**: Just add three kwargs to your standard Bilby run

## Installation

The flow packages are optional extras; install the one(s) you train with.

### Standard Installation

```bash
git clone https://github.com/mrosep/bilby-pr.git
cd bilby-pr
pip install ".[margarine_unbounded]"   # or ".[margarine]", or ".[margarine,margarine_unbounded]"
```

**Quick install from GitHub:**
```bash
pip install "bilby-pr[margarine_unbounded] @ git+https://github.com/mrosep/bilby-pr.git"
```

### Development Installation

```bash
cd /path/to/bilby-pr
pip install -e ".[margarine_unbounded]"
```

### Keras 3

Both flow packages need Keras 2 (`tf_keras`, installed as a dependency). On import, bilby-pr sets `TF_USE_LEGACY_KERAS=1`, logging a warning if it overrides a different value. This only takes effect if tensorflow has not been imported before bilby-pr; if you import tensorflow (or a flow package) first in your script, set `TF_USE_LEGACY_KERAS=1` yourself.

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
    flow_backend='margarine_unbounded',        # or 'margarine'
    nlive=500,
    npool=4,                                    # Multiprocessing supported
    # ... other standard Dynesty kwargs
)
```

### Training a Flow

Before using posterior repartitioning, you need to train a normalizing flow on a preliminary guess at the posterior (e.g., samples from a quick initial run, approximate samples, or samples from a similar problem).

Example training workflow with margarine_unbounded (`flow_backend='margarine_unbounded'`):
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

Example training workflow with margarine (`flow_backend='margarine'`):
```python
import numpy as np
from margarine.maf import MAF

samples = ...  # Shape: (n_samples, n_parameters), columns in flow_params order

# Bounds of the flow; if omitted, margarine estimates them from the samples.
# The flow can only produce values inside this box.
theta_min = np.array([priors[key].minimum for key in flow_params])
theta_max = np.array([priors[key].maximum for key in flow_params])

maf = MAF(samples, number_networks=10, hidden_layers=[128, 128],
          theta_min=theta_min, theta_max=theta_max)
maf.train(epochs=100)
maf.save('trained_flow.pkl')
```

In both cases the standardisation (`mean`/`std`) or bounds (`theta_min`/`theta_max`) are stored in the `.pkl` file and restored when bilby-pr loads the flow.

## How it Works

### 1. Prior Transform

For flow-modeled parameters:
```
Uniform [0,1] → Standard Normal N(0,1) → MAF → Physical Parameters
                    (via quantile function)
```

For `margarine`, the MAF output is additionally mapped into the box `[theta_min, theta_max]`; for `margarine_unbounded` it is unstandardised with the training mean and std.

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
- bilby >= 2.7.0
- margarine (>=1.2.8, <2) or margarine_unbounded
- tensorflow >= 2.8.0
- tensorflow-probability >= 0.16.0
- tf_keras
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
4. `flow_backend` matches the package the flow was trained with

For `margarine` flows, the sampler can never propose flow parameters outside `[theta_min, theta_max]`. If the posterior has mass outside that box, it is silently truncated, so set the bounds wide enough when training (e.g. to the prior bounds).

### TensorFlow Warnings

You may see TensorFlow warnings about CPU features or threading. These are usually harmless and can be ignored. The sampler explicitly configures TensorFlow for CPU-only operation to ensure stability with multiprocessing.

## Citation

If you use this package, please cite:
- The Bilby paper: [Ashton et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJS..241...27A)
- The margarine paper: [Bevins et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.2923B)
- Posterior repartitioning for GWs papers: [Prathaban et al. 2025](https://academic.oup.com/mnras/article/541/1/200/8163830) and [add arXiv link for simple-pe-PR]()

## License

MIT License - see LICENSE file for details
