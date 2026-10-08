from bilby.core.sampler.dynesty import Dynesty
from bilby.core.sampler.base_sampler import signal_wrapper
from bilby.core.likelihood import _safe_likelihood_call
from unittest.mock import patch
import numpy as np
import tensorflow as tf

from .utils import PRGlobalVariablesMixin


def _prior_transform_wrapper(theta):
    """Transform uniform [0,1] samples to physical parameter space.

    This wrapper is needed for multiprocessing compatibility with Bilby's Dynesty sampler.

    For parameters modeled by the flow:
        - Uses the flow adapter's .quantile() method (see flows.py)
        - Transforms: uniform [0,1] → standard normal → MAF → physical parameters
          (margarine additionally maps the output into its [theta_min, theta_max] box)

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
    y = _sampling_convenience_dump.maf_model_quantile(theta_scale)

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

    search_params = {
        key: t
        for key, t in zip(_sampling_convenience_dump.search_parameter_keys, theta)
    }

    # Compute original prior probability for ALL search parameters
    prior_logprob = _sampling_convenience_dump.priors.ln_prob(search_params)

    # Likelihood also needs the fixed parameters (e.g. marginalised distance)
    params = {**_sampling_convenience_dump.parameters, **search_params}

    if np.isfinite(prior_logprob):
        # Extract flow-modeled parameters
        theta_scale = np.array([theta[i] for i in _sampling_convenience_dump.flow_params_indices])

        # Compute flow density q(θ_flow) for flow parameters
        maf_logprob = _sampling_convenience_dump.maf_model_prob(theta_scale)

        if np.isfinite(maf_logprob):
            # Compute prior probability for non-flow parameters: π(θ_non-flow)
            prior_correction = 0
            for i, key in enumerate(_sampling_convenience_dump.search_parameter_keys):
                if key not in _sampling_convenience_dump.flow_params:
                    prior_correction += _sampling_convenience_dump.priors[key].ln_prob(theta[i])

            # Compute likelihood
            logL = _safe_likelihood_call(
                _sampling_convenience_dump.likelihood,
                params,
                _sampling_convenience_dump.use_ratio,
            )

            # Return: log[L(θ) × π(θ) / q_new(θ)]
            #       = logL + log[π(θ)] - log[q(θ_flow)] - log[π(θ_non-flow)]
            #       = logL + prior_logprob - maf_logprob - prior_correction
            return logL + prior_logprob - maf_logprob - prior_correction

    # If we reach here, either prior or flow probability was not finite
    return np.nan_to_num(-np.inf)

class DynestyPR(PRGlobalVariablesMixin, Dynesty):
    """Dynesty nested sampler with posterior repartitioning using normalizing flows.

    This sampler extends Bilby's standard Dynesty sampler to use trained normalizing flows
    (MAFs from margarine or margarine_unbounded) to repartition the prior during
    nested sampling.

    Key features:
        - Transforms selected parameters using trained flows via the backend's
          uniform-to-physical transform
        - Other parameters use standard Bilby priors
        - Modifies likelihood to account for the change of sampling prior
        - Accelerates sampling by focusing on high-probability regions

    Required kwargs:
        weights_file (str): Path to the trained MAF model (.pkl file)
        flow_params (list): List of parameter names to model with the flow
            (must be in the same order as used during training)
        flow_backend (str): Package the flow was trained with,
            'margarine' or 'margarine_unbounded'

    Example:
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler='dynesty_pr',
            weights_file='trained_flow.pkl',
            flow_params=['mass_ratio', 'chirp_mass', 'theta_jn'],
            flow_backend='margarine_unbounded',
            nlive=500
        )
    """

    sampler_name = "dynesty_pr"

    @property
    def external_sampler_name(self) -> str:
        """The name of the package that provides this sampler."""
        return "bilby_pr"

    def get_initial_points_from_prior(self, npoints=1):
        """Draw the initial live points from the repartitioned prior.

        Overrides Bilby's version, which draws from the original prior and stores the
        unmodified likelihood. This mirrors it, but uses the flow-based prior transform
        and the modified likelihood, so the initial live points are consistent with
        the rest of the run.

        Parameters
        ==========
        npoints: int
            The number of values to return

        Returns
        =======
        unit_cube, parameters, likelihood: tuple of array_like
            unit_cube (nlive, ndim) is an array of the samples from the
            unit cube, parameters (nlive, ndim) is the unit_cube array
            transformed to the target space, while likelihood (nlive) are the
            modified likelihood evaluations.
        """
        from bilby.core.utils import logger, random

        logger.info("Generating initial points from the flow (posterior repartitioning)")
        unit_cube = []
        parameters = []
        likelihood = []
        while len(unit_cube) < npoints:
            unit = random.rng.uniform(0, 1, self.ndim)
            theta = _prior_transform_wrapper(unit)
            if self._check_draw_pr(theta, warning=False):
                unit_cube.append(unit)
                parameters.append(theta)
                likelihood.append(_log_likelihood_wrapper(theta))

        return np.array(unit_cube), np.array(parameters), np.array(likelihood)

    def _check_draw_pr(self, theta, warning=True):
        """Bilby's check_draw, but with the modified likelihood.

        Kept separate from check_draw, which Bilby also calls while constructing
        the sampler, before the flow has been loaded.
        """
        log_p = self.log_prior(theta)
        log_l = _log_likelihood_wrapper(theta)
        return self._check_bad_value(
            val=log_p, warning=warning, theta=theta, label="prior"
        ) and self._check_bad_value(
            val=log_l, warning=warning, theta=theta, label="likelihood"
        )

    @signal_wrapper
    def run_sampler(self):
        """Run the Dynesty sampler with posterior repartitioning.

        Patches Bilby's standard wrappers to use the flow-based transformations
        and modified likelihood computation.
        """
        with patch("bilby.core.sampler.dynesty._log_likelihood_wrapper", _log_likelihood_wrapper), \
                patch("bilby.core.sampler.dynesty._prior_transform_wrapper", _prior_transform_wrapper):
            return super().run_sampler()
