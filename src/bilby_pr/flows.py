"""Adapters giving every flow backend the same interface.

Each backend wraps a trained flow from a different package and exposes:
    - load(weights_file): load the trained flow from file
    - quantile(u): map uniform [0,1] samples to physical parameters
    - log_prob(x): log-density of the flow at physical parameters x

Outputs are returned as numpy arrays so that backends built on different
frameworks (TensorFlow now, JAX later) look the same to the sampler.
"""
from abc import ABC, abstractmethod

import numpy as np


class FlowAdapter(ABC):
    """Common interface to a trained flow, whichever package it was trained with."""

    def __init__(self, flow):
        self.flow = flow

    @classmethod
    @abstractmethod
    def load(cls, weights_file):
        """Load a trained flow from file."""

    @abstractmethod
    def quantile(self, u):
        """Transform uniform [0,1] samples to physical parameters."""

    @abstractmethod
    def log_prob(self, x):
        """Log-probability of the flow at physical parameters x."""


class MargarineFlow(FlowAdapter):
    """MAF from margarine (1.x, TensorFlow).

    The flow is bounded to [theta_min, theta_max], which are set when training
    and restored from the weights file by MAF.load().
    """

    @classmethod
    def load(cls, weights_file):
        from margarine.maf import MAF

        return cls(MAF.load(weights_file))

    def quantile(self, u):
        return np.asarray(self.flow(u))

    def log_prob(self, x):
        return np.asarray(self.flow.log_prob(x))


class MargarineUnboundedFlow(FlowAdapter):
    """MAF from margarine_unbounded (TensorFlow).

    The flow is unbounded; it standardises parameters with the mean and std
    of the training samples, which are restored from the weights file by MAF.load().
    """

    @classmethod
    def load(cls, weights_file):
        from margarine_unbounded.maf import MAF

        return cls(MAF.load(weights_file))

    def quantile(self, u):
        return np.asarray(self.flow.quantile(u))

    def log_prob(self, x):
        return np.asarray(self.flow.log_prob(x))


FLOW_BACKENDS = {
    "margarine": MargarineFlow,
    "margarine_unbounded": MargarineUnboundedFlow,
}
