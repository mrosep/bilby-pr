from bilby.core.sampler.dynesty import Dynesty
from bilby.core.sampler.base_sampler import signal_wrapper
from unittest.mock import patch
import numpy as np

from .utils import PRGlobalVariablesMixin


def _prior_transform_wrapper(theta):
    """Wrapper to the prior transformation. Needed for multiprocessing."""
    from .utils import _sampling_convenience_dump

    theta_scale = np.array([theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[8], theta[9]])
    y = _sampling_convenience_dump.maf_model(theta_scale).numpy()

    # Explicitly scale psi, phase and time according to the prior
    psi_transform = _sampling_convenience_dump.priors['psi'].rescale(theta[6])
    phase_transform = _sampling_convenience_dump.priors['phase'].rescale(theta[7])
    time_transform = _sampling_convenience_dump.priors['geocent_time'].rescale(theta[10])

    rescaled = np.array([y[0], y[1], y[2], y[3], y[4], y[5], psi_transform, phase_transform, y[6], y[7], time_transform])
    return rescaled


# @profile
def _log_likelihood_wrapper(theta):
    """Wrapper to the log likelihood. Needed for multiprocessing."""
    from .utils import _sampling_convenience_dump

    params = {
        key: t
        for key, t in zip(_sampling_convenience_dump.search_parameter_keys, theta)
    }

    prior_logprob = _sampling_convenience_dump.priors.ln_prob(params)
    if np.isfinite(prior_logprob):
        _sampling_convenience_dump.likelihood.parameters.update(params)

        # TODO: this needs to be more robust to the order of the parameters

        theta_scale = np.array([theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[8], theta[9]])
        # Is there way to disable the gradients to speed things up?
        maf_logprob = _sampling_convenience_dump.maf_model_prob(theta_scale).numpy()
        prior_correction = _sampling_convenience_dump.priors['psi'].ln_prob(theta[6]) + _sampling_convenience_dump.priors['phase'].ln_prob(theta[7]) + _sampling_convenience_dump.priors['geocent_time'].ln_prob(theta[10])

        # Return L*pi/new_prior_prob
        if _sampling_convenience_dump.use_ratio:
            logL = _sampling_convenience_dump.likelihood.log_likelihood_ratio()
        else:
            logL = _sampling_convenience_dump.likelihood.log_likelihood()
        return logL + prior_logprob - maf_logprob - prior_correction
    else:
        return np.nan_to_num(-np.inf)


class DynestyPR(PRGlobalVariablesMixin, Dynesty):
    """A Dynesty sampler with support for posterior repartitioning"""

    sampler_name = "dynesty_pr"

    @property
    def external_sampler_name(self) -> str:
        """The name of package that provides the sampler."""
        return "bilby_pr"

    @signal_wrapper
    def run_sampler(self):
        # We need to patch the log likelihood and prior transform wrappers
        # to use the global variables set up in the PRGlobalVariablesMixin
        # Rather than the standard ones from bilby
        with patch("bilby.core.sampler.dynesty._log_likelihood_wrapper", _log_likelihood_wrapper), \
                patch("bilby.core.sampler.dynesty._prior_transform_wrapper", _prior_transform_wrapper):
            return super().run_sampler()
