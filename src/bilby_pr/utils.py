from bilby.core.utils import logger
from bilby.core.sampler.base_sampler import _SamplingContainer
from bilby.core.utils import random
import os
import tensorflow as tf
from margarine_unbounded.maf import MAF


_sampling_convenience_dump = _SamplingContainer()


def _initialize_global_variables(
    likelihood,
    priors,
    search_parameter_keys,
    use_ratio,
    weights_file,
    flow_params,
):
    """Initialize global variables for posterior repartitioning in worker processes.

    This function is called once in each worker process (or in the main process if not
    using multiprocessing) to set up the trained MAF model and related state.

    Args:
        likelihood: Bilby likelihood object
        priors: Bilby prior dictionary
        search_parameter_keys: List of parameter names being sampled
        use_ratio: Whether to use likelihood ratio (Bilby setting)
        weights_file: Path to trained MAF model (.pkl file)
        flow_params: List of parameter names to be modeled by the flow
    """
    global _sampling_convenience_dump
    _sampling_convenience_dump.likelihood = likelihood
    _sampling_convenience_dump.priors = priors
    _sampling_convenience_dump.search_parameter_keys = search_parameter_keys
    _sampling_convenience_dump.use_ratio = use_ratio

    # Map flow parameter names to their indices in the full parameter list
    flow_params_indices = [search_parameter_keys.index(param) for param in flow_params]

    _sampling_convenience_dump.flow_params = flow_params
    _sampling_convenience_dump.flow_params_indices = flow_params_indices

    # Configure TensorFlow for CPU-only operation with minimal threading
    # This prevents GPU memory conflicts and improves multiprocessing stability
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.keras.backend.clear_session()

    # Load the trained MAF model
    logger.info(f"Loading weights from: {weights_file}")
    _sampling_convenience_dump.maf_model = MAF.load(weights_file)
    _sampling_convenience_dump.maf_model_prob = _sampling_convenience_dump.maf_model.log_prob
    _sampling_convenience_dump.maf_model_quantile = _sampling_convenience_dump.maf_model.quantile

    # Warm up the flow with test evaluations (compiles TensorFlow graphs)
    print("maf_model quantile", _sampling_convenience_dump.maf_model_quantile(random.rng.uniform(0, 1, size=len(flow_params))))
    print("maf_model_prob", _sampling_convenience_dump.maf_model_prob(random.rng.uniform(0, 1, size=len(flow_params))))


class PRGlobalVariablesMixin:
    """Mixin class to enable posterior repartitioning in Bilby samplers.

    This mixin extends Bilby samplers to support loading and using trained normalizing
    flows for prior repartitioning. It handles multiprocessing pool setup and ensures
    that each worker process has access to the flow model.

    This should work with any Bilby sampler that uses multiprocessing.

    Adds two required kwargs to the sampler:
        weights_file: Path to trained MAF model
        flow_params: List of parameter names to model with the flow
    """

    @property
    def weights_file(self):
        """Path to the trained MAF model file (.pkl)."""
        return self.kwargs.get("weights_file", None)

    @property
    def flow_params(self):
        """List of parameter names to be modeled by the flow."""
        return self.kwargs.get("flow_params", None)

    @property
    def default_kwargs(self):
        """Add PR-specific kwargs to the sampler's default kwargs."""
        kwargs = super().default_kwargs
        kwargs["weights_file"] = None
        kwargs['flow_params'] = None
        return kwargs

    def _setup_pool(self):
        """Set up multiprocessing pool with posterior repartitioning initialization.

        Overrides Bilby's standard _setup_pool() to ensure that worker processes
        are initialized with the trained flow model and related state.
        """
        if self.kwargs.get("pool", None) is not None:
            logger.info("Using user defined pool.")
            self.pool = self.kwargs["pool"]
        elif self.npool is not None and self.npool > 1:
            logger.info(f"Setting up multiprocessing pool with {self.npool} processes")
            import multiprocessing

            self.pool = multiprocessing.Pool(
                processes=self.npool,
                initializer=_initialize_global_variables,
                initargs=(
                    self.likelihood,
                    self.priors,
                    self._search_parameter_keys,
                    self.use_ratio,
                    self.weights_file,
                    self.flow_params,
                ),
            )
        else:
            self.pool = None

        # Initialize global variables in the main process as well
        _initialize_global_variables(
            likelihood=self.likelihood,
            priors=self.priors,
            search_parameter_keys=self._search_parameter_keys,
            use_ratio=self.use_ratio,
            weights_file=self.weights_file,
            flow_params=self.flow_params,
        )
        self.kwargs["pool"] = self.pool
