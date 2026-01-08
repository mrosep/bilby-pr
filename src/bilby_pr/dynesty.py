from bilby.core.sampler.dynesty import Dynesty
from bilby.core.sampler.base_sampler import signal_wrapper
from unittest.mock import patch
import numpy as np
import tensorflow as tf

from .utils import PRGlobalVariablesMixin


def _prior_transform_wrapper(theta):
    """Transform uniform [0,1] samples to physical parameter space.

    This wrapper is needed for multiprocessing compatibility with Bilby's Dynesty sampler.

    For parameters modeled by the flow:
        - Uses margarine_unbounded's .quantile() method
        - Transforms: uniform [0,1] → standard normal → MAF → physical parameters

    For other parameters:
        - Uses Bilby's standard prior.rescale() method

    Args:
        theta: Array of uniform [0,1] samples, one per parameter

    Returns:
        Array of samples in physical parameter space
    """
    from .utils import _sampling_convenience_dump

    rescaled = np.zeros(len(_sampling_convenience_dump.search_parameter_keys))

    # Extract uniform samples for flow-modeled parameters
    theta_scale = np.array([theta[i] for i in _sampling_convenience_dump.flow_params_indices])

    # Transform uniform samples to physical space via the flow
    y = _sampling_convenience_dump.maf_model_quantile(theta_scale).numpy()

    # Create mapping from global parameter index to flow parameter index
    flow_index_mapping = {
        _sampling_convenience_dump.flow_params_indices[i]: i
        for i in range(len(_sampling_convenience_dump.flow_params_indices))
    }

    # Fill in the rescaled parameter array
    for i in range(len(_sampling_convenience_dump.search_parameter_keys)):
        if i not in _sampling_convenience_dump.flow_params_indices:
            # Non-flow parameters: use standard Bilby prior transformation
            rescaled[i] = _sampling_convenience_dump.priors[
                _sampling_convenience_dump.search_parameter_keys[i]
            ].rescale(theta[i])
        else:
            # Flow parameters: use the quantile-transformed values
            rescaled[i] = y[flow_index_mapping[i]]

    return rescaled


def _log_likelihood_wrapper(theta):
    """Compute the modified log-likelihood for posterior repartitioning.

    This wrapper is needed for multiprocessing compatibility with Bilby's Dynesty sampler.

    Computes the modified likelihood:
        L_modified(θ) = L(θ) × π(θ) / q_new(θ)

    where:
        - π(θ) is the original prior for ALL parameters
        - q_new(θ) is the repartitioned prior = q(θ_flow) × π(θ_non-flow)
          - q(θ_flow): flow density for flow-modeled parameters
          - π(θ_non-flow): original prior for non-flow parameters

    Args:
        theta: Array of parameter values in physical space

    Returns:
        Modified log-likelihood, or -inf if outside prior or flow support
    """
    from .utils import _sampling_convenience_dump

    params = {
        key: t
        for key, t in zip(_sampling_convenience_dump.search_parameter_keys, theta)
    }

    # Compute original prior probability for ALL parameters
    prior_logprob = _sampling_convenience_dump.priors.ln_prob(params)
    _sampling_convenience_dump.likelihood.parameters.update(params)

    if np.isfinite(prior_logprob):
        # Extract flow-modeled parameters
        theta_scale = np.array([theta[i] for i in _sampling_convenience_dump.flow_params_indices])

        # Compute flow density q(θ_flow) for flow parameters
        maf_logprob = _sampling_convenience_dump.maf_model_prob(theta_scale).numpy()

        if np.isfinite(maf_logprob):
            # Compute prior probability for non-flow parameters: π(θ_non-flow)
            prior_correction = 0
            for i, key in enumerate(_sampling_convenience_dump.search_parameter_keys):
                if key not in _sampling_convenience_dump.flow_params:
                    prior_correction += _sampling_convenience_dump.priors[key].ln_prob(theta[i])

            # Compute likelihood
            if _sampling_convenience_dump.use_ratio:
                logL = _sampling_convenience_dump.likelihood.log_likelihood_ratio()
            else:
                logL = _sampling_convenience_dump.likelihood.log_likelihood()

            # Return: log[L(θ) × π(θ) / q_new(θ)]
            #       = logL + log[π(θ)] - log[q(θ_flow)] - log[π(θ_non-flow)]
            #       = logL + prior_logprob - maf_logprob - prior_correction
            return logL + prior_logprob - maf_logprob - prior_correction

    # If we reach here, either prior or flow probability was not finite
    return np.nan_to_num(-np.inf)

class DynestyPR(PRGlobalVariablesMixin, Dynesty):
    """Dynesty nested sampler with posterior repartitioning using normalizing flows.

    This sampler extends Bilby's standard Dynesty sampler to use trained normalizing flows
    (MAFs from margarine_unbounded) to repartition the prior during nested sampling.

    Key features:
        - Transforms selected parameters using trained flows via .quantile() method
        - Other parameters use standard Bilby priors
        - Modifies likelihood to account for the change of sampling prior
        - Accelerates sampling by focusing on high-probability regions

    Required kwargs:
        weights_file (str): Path to the trained MAF model (.pkl file)
        flow_params (list): List of parameter names to model with the flow
            (must be in the same order as used during training)

    Example:
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler='dynesty_pr',
            weights_file='trained_flow.pkl',
            flow_params=['mass_ratio', 'chirp_mass', 'theta_jn'],
            nlive=500
        )
    """

    sampler_name = "dynesty_pr"

    @property
    def external_sampler_name(self) -> str:
        """The name of the package that provides this sampler."""
        return "bilby_pr"

    @signal_wrapper
    def run_sampler(self):
        """Run the Dynesty sampler with posterior repartitioning.

        Patches Bilby's standard wrappers to use the flow-based transformations
        and modified likelihood computation.
        """
        with patch("bilby.core.sampler.dynesty._log_likelihood_wrapper", _log_likelihood_wrapper), \
                patch("bilby.core.sampler.dynesty._prior_transform_wrapper", _prior_transform_wrapper):
            return super().run_sampler()
