import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.distribution import Distribution
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import ExpTransform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.utils import (
    broadcast_all,
    clamp_probs,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)

import numpy as np
import tqdm
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.ops.indexing import Vindex
from pyro.distributions.torch_distribution import TorchDistributionMixin
import scanpy as sc
import math
from pyro.distributions.torch import RelaxedBernoulli, RelaxedOneHotCategorical

class SafeExpRelaxedCategorical(ExpRelaxedCategorical):
    #Thanks @fritzo
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        uniforms = clamp_probs(
            torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device)
        )
        gumbels = -((-(uniforms.log())).log())
        scores = (self.logits + gumbels) / self.temperature
        #could also clamp_probs
        outs = scores - scores.logsumexp(dim=-1, keepdim=True)
        outs = outs.exp()
        outs = outs.clamp(min=torch.finfo(outs.dtype).tiny)
        outs = (outs / outs.sum(1, keepdim=True)).log()
        return outs


class SafeAndRelaxedOneHotCategorical(TransformedDistribution,TorchDistributionMixin):
    r"""
    Creates a RelaxedOneHotCategorical distribution parametrized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits`.
    This is a relaxed version of the :class:`OneHotCategorical` distribution, so
    its samples are on simplex, and are reparametrizable.
    https://github.com/pytorch/pytorch/blob/main/torch/distributions/relaxed_categorical.py
    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = RelaxedOneHotCategorical(torch.tensor([2.2]),
        ...                              torch.tensor([0.1, 0.2, 0.3, 0.4]))
        >>> m.sample()
        tensor([ 0.1294,  0.2324,  0.3859,  0.2523])

    Args:
        temperature (Tensor): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event
    """
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        base_dist = SafeExpRelaxedCategorical(
            temperature, probs, logits, validate_args=validate_args
        )
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SafeAndRelaxedOneHotCategorical, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def temperature(self):
        return self.base_dist.temperature

    @property
    def logits(self):
        return self.base_dist.logits

    @property
    def probs(self):
        return self.base_dist.probs

class SafeAndRelaxedOneHotCategoricalStraightThrough(RelaxedOneHotCategorical):
    def rsample(self, sample_shape=torch.Size()):
        soft_sample = super().rsample(sample_shape)
        soft_sample = clamp_probs(soft_sample)
        hard_sample = QuantizeCategorical.apply(soft_sample)
        hard_sample = hard_sample.clamp(min=torch.finfo(hard_sample.dtype).tiny)
        #hard_sample = (hard_sample / hard_sample.sum(1, keepdim=True))
        return hard_sample

    def log_prob(self, value):
        value = getattr(value, "_unquantize", value)
        return super().log_prob(value)


class QuantizeCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft_value):
        argmax = soft_value.max(-1)[1]
        hard_value = torch.zeros_like(soft_value)
        hard_value._unquantize = soft_value
        if argmax.dim() < hard_value.dim():
            argmax = argmax.unsqueeze(-1)
        return hard_value.scatter_(-1, argmax, 1)

    @staticmethod
    def backward(ctx, grad):
        return grad
