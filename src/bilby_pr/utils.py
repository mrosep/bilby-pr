from bilby.core.utils import logger
from bilby.core.sampler.base_sampler import _SamplingContainer
from bilby.core.utils import random
import os
import tensorflow as tf
from margarine.maf import MAF


_sampling_convenience_dump = _SamplingContainer()


def _initialize_global_variables(
    likelihood,
    priors,
    search_parameter_keys,
    use_ratio,
    weights_file,
):
    """Initialize global variables for posterior repartitioning.

    This includes the weights file for the MAF model.
    """
    global _sampling_convenience_dump
    _sampling_convenience_dump.likelihood = likelihood
    _sampling_convenience_dump.priors = priors
    _sampling_convenience_dump.search_parameter_keys = search_parameter_keys
    _sampling_convenience_dump.use_ratio = use_ratio

    # Manually control threading and GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "" # disable GPU
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    # Clear any previously allocated TF state in the child processes
    tf.keras.backend.clear_session()

    logger.info(f"Loading weights from: {weights_file}")
    _sampling_convenience_dump.maf_model = MAF.load(weights_file)
    _sampling_convenience_dump.maf_model_prob = _sampling_convenience_dump.maf_model.log_prob

    # Warm up flow
    # TODO: Probably need way to figure out the dimensionality of the flow
    print("maf_model", _sampling_convenience_dump.maf_model(random.rng.uniform(0, 1, size=8)))
    print("maf_model_prob", _sampling_convenience_dump.maf_model_prob(random.rng.uniform(0, 1, size=8)))


class PRGlobalVariablesMixin:
    """Mixing to set up global variables for posterior repartitioning.

    This should work with any sampler that uses multiprocessing from bilby.
    """

    @property
    def weights_file(self):
        return self.kwargs.get("weights_file", None)

    @property
    def default_kwargs(self):
        kwargs = super().default_kwargs
        kwargs["weights_file"] = None
        return kwargs

    def _setup_pool(self):
        """Version of _setup_pool that sets up global variables for posterior repartitioning."""
        if self.kwargs.get("pool", None) is not None:
            logger.info("Using user defined pool.")
            self.pool = self.kwargs["pool"]
        elif self.npool is not None and self.npool > 1:
            logger.info(f"Setting up multiproccesing pool with {self.npool} processes")
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
                ),
            )
        else:
            self.pool = None
        _initialize_global_variables(
            likelihood=self.likelihood,
            priors=self.priors,
            search_parameter_keys=self._search_parameter_keys,
            use_ratio=self.use_ratio,
            weights_file=self.weights_file,
        )
        self.kwargs["pool"] = self.pool
