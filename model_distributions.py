import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.distribution import Distribution
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


class ExpRelaxedCategorical(Distribution):
    r"""
    Creates a ExpRelaxedCategorical parameterized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits` (but not both).
    Returns the log of a point in the simplex. Based on the interface to
    :class:`OneHotCategorical`.

    Implementation based on [1].

    See also: :func:`torch.distributions.OneHotCategorical`

    Args:
        temperature (float): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
    (Maddison et al, 2017)

    [2] Categorical Reparametrization with Gumbel-Softmax
    (Jang et al, 2017)
    """
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    support = (
        constraints.real_vector
    )  # The true support is actually a submanifold of this.
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        self._categorical = Categorical(probs, logits)
        self.temperature = temperature
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ExpRelaxedCategorical, _instance)
        batch_shape = torch.Size(batch_shape)
        new.temperature = self.temperature
        new._categorical = self._categorical.expand(batch_shape)
        super(ExpRelaxedCategorical, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    @property
    def param_shape(self):
        return self._categorical.param_shape

    @property
    def logits(self):
        return self._categorical.logits

    @property
    def probs(self):
        return self._categorical.probs

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        uniforms = clamp_probs(
            torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device)
        )
        gumbels = -((-(uniforms.log())).log())
        scores = (self.logits + gumbels) / self.temperature
        outs=scores - scores.logsumexp(dim=-1, keepdim=True)
        outs=outs.exp()
        outs=outs+1e-10
        outs=(outs/outs.sum(1,keepdim=True)).log()
        return outs

    def log_prob(self, value):
        K = self._categorical._num_events
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        log_scale = torch.full_like(
            self.temperature, float(K)
        ).lgamma() - self.temperature.log().mul(-(K - 1))
        score = logits - value.mul(self.temperature)
        score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)
        return score + log_scale

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
        base_dist = ExpRelaxedCategorical(
            temperature, probs, logits, validate_args=validate_args
        )
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RelaxedOneHotCategorical, _instance)
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


'''
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
        
class ExpCategoricalStraightThrough(Distribution):
    r"""
    Creates a ExpRelaxedCategorical parameterized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits` (but not both).
    Returns the log of a point in the simplex. Based on the interface to
    :class:`OneHotCategorical`.

    Implementation based on [1].

    See also: :func:`torch.distributions.OneHotCategorical`

    Args:
        temperature (Tensor): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
    (Maddison et al, 2017)

    [2] Categorical Reparametrization with Gumbel-Softmax
    (Jang et al, 2017)
    """
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    support = (
        constraints.real_vector
    )  # The true support is actually a submanifold of this.
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        self._categorical = Categorical(probs, logits)
        self.temperature = temperature/temperature#always 1
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ExpRelaxedCategoricalStraightThrough, _instance)
        batch_shape = torch.Size(batch_shape)
        new.temperature = self.temperature
        new._categorical = self._categorical.expand(batch_shape)
        super(ExpRelaxedCategoricalStraightThrough, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    @property
    def param_shape(self):
        return self._categorical.param_shape

    @property
    def logits(self):
        return self._categorical.logits

    @property
    def probs(self):
        return self._categorical.probs

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        scores=self.logits
        outs=scores - scores.logsumexp(dim=-1, keepdim=True)
        outs=outs.exp()
        hard_sample = QuantizeCategorical.apply(outs)
        hard_sample=hard_sample+1e-10
        hard_sample=(hard_sample/hard_sample.sum(1,keepdim=True)).log()
        return hard_sample

    def log_prob(self, value):
        value = getattr(value, "_unquantize", value)
        K = self._categorical._num_events
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        score = logits 
        score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)
        return score 


class SafeOneHotCategoricalStraightThrough(TransformedDistribution,TorchDistributionMixin):
    #Don't understand why these were broken (doesn't call straighthrough rsample in pyro)?
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        base_dist = ExpRelaxedCategoricalStraightThrough(
            temperature, probs, logits, validate_args=validate_args
        )
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RelaxedOneHotCategorical, _instance)
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
'''

class RelaxedQuantizeCategorical(torch.autograd.Function):
    temperature = None  # Default temperature
    epsilon = 1e-10    # Default epsilon

    @staticmethod
    def set_temperature(new_temperature):
        RelaxedQuantizeCategorical.temperature = new_temperature

    @staticmethod
    def set_epsilon(new_epsilon):
        RelaxedQuantizeCategorical.epsilon = new_epsilon

    @staticmethod
    def forward(ctx, soft_value):
        temperature = float(RelaxedQuantizeCategorical.temperature)
        epsilon = RelaxedQuantizeCategorical.epsilon
        uniforms = clamp_probs(
            torch.rand(soft_value.shape, dtype=soft_value.dtype, device=soft_value.device)
        )
        gumbels = -((-(uniforms.log())).log())
        scores = (soft_value + gumbels) / temperature
        outs = scores - scores.logsumexp(dim=-1, keepdim=True)
        outs = outs.exp()
        outs = outs + epsilon  # Use the class variable epsilon
        hard_value = (outs / outs.sum(1, keepdim=True)).log()
        hard_value._unquantize = soft_value
        return hard_value

    @staticmethod
    def backward(ctx, grad):
        return grad


class ExpRelaxedCategoricalStraightThrough(Distribution):
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    support = (
        constraints.real_vector
    )  # The true support is actually a submanifold of this.
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None, epsilon=1e-10):
        self._categorical = Categorical(probs, logits)
        self.temperature = temperature
        RelaxedQuantizeCategorical.set_temperature(temperature)
        RelaxedQuantizeCategorical.set_epsilon(epsilon)
        
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ExpRelaxedCategorical, _instance)
        batch_shape = torch.Size(batch_shape)
        new.temperature = self.temperature
        new._categorical = self._categorical.expand(batch_shape)
        super(ExpRelaxedCategorical, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    @property
    def param_shape(self):
        return self._categorical.param_shape

    @property
    def logits(self):
        return self._categorical.logits

    @property
    def probs(self):
        return self._categorical.probs

    def rsample(self, sample_shape=torch.Size()):
        outs=RelaxedQuantizeCategorical.apply(self.logits)
        return outs

    def log_prob(self, value):
        value = getattr(value, "_unquantize", value)
        K = self._categorical._num_events
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        score = logits 
        score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)
        return score 

class SafeAndRelaxedOneHotCategoricalStraightThrough(TransformedDistribution,TorchDistributionMixin):
    #Don't understand why these were broken (doesn't call straighthrough rsample in pyro)?
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        base_dist = ExpRelaxedCategoricalStraightThrough(
            temperature, probs, logits, validate_args=validate_args
        )
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RelaxedOneHotCategorical, _instance)
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



from numbers import Number
from numbers import Real
class TorchGeneralizedNormal(torch.distributions.exp_family.ExponentialFamily):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'beta': constraints.positive}
    support = constraints.real
    has_rsample = False # not implemented
    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        # Using the formula for the variance of a Generalized Normal Distribution
        return (self.scale ** 2) * (math.gamma(3 / self.beta) / math.gamma(1 / self.beta))        

    def __init__(self, loc, scale, beta, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        self.beta=beta
        if isinstance(loc, Number) and isinstance(scale, Number) and isinstance(beta, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=False)
    
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TorchGeneralizedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.beta = self.beta
        super(TorchGeneralizedNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new  
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -(1 / self.beta) * torch.log(self.scale) - torch.log(2 * torch.exp(torch.lgamma(1. / self.beta))) - torch.pow(torch.abs(value - self.loc), self.beta) / torch.pow(self.scale, self.beta)
        #return -(torch.pow(torch.abs((value - self.loc) / self.scale), self.beta) - torch.log(self.scale * self.beta * torch.exp(torch.lgamma(1 / self.beta))))

    def _cdf_zero_mean(self, x):
        zero = torch.tensor(0., dtype=self.scale.dtype)
        half = torch.tensor(0.5, dtype=self.scale.dtype)
        one = torch.tensor(1., dtype=self.scale.dtype)

        x_is_zero = (x == zero)
        safe_x = torch.where(x_is_zero, one, x)
        half_gamma = half * torch.exp(torch.lgamma(1/self.beta) - torch.pow(torch.abs(safe_x) / self.scale, self.beta))

        return torch.where(x_is_zero, half, torch.where(x > zero, one - half_gamma, half_gamma))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self._cdf_zero_mean(value - self.loc)

    def _survival_function(self, x):
        # sf(x) = cdf(-x) for loc == 0, because distribution is symmetric.
        return self._cdf_zero_mean(self.loc - x)

    def _quantile(self, p):
        ipower = 1/self.beta
        quantile = torch.where(
            p < 0.5,
            self.loc - torch.pow(torch.special.gammaincc(ipower, 2. * p), ipower) * self.scale,
            self.loc + torch.pow(torch.special.gammainc(ipower, 2. * p - 1.), ipower) * self.scale)
        return quantile
    
    def _rademacher(self, shape):
        return (torch.rand(shape) > 0.5).float() * 2 - 1

    def _sample_n(self, n, seed=None):
        n = torch.tensor(n, dtype=torch.int32)
        batch_shape = torch.Size((n.item(),) + self.loc.size())

        ipower = 1/self.beta.expand(batch_shape)
        gamma_dist = Gamma(ipower, 1.)

        gamma_sample = gamma_dist.sample((n.item(),))
        binary_sample = self._rademacher(gamma_sample.shape)
        sampled = (binary_sample * torch.pow(torch.abs(gamma_sample), ipower))

        return self.loc + self.scale * sampled

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self._sample_n(sample_shape.numel())
        
    def _natural_params(self):
        return (self.loc / self.scale.pow(self.beta), -0.5 * self.scale.pow(self.beta).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(self.beta) / y + 0.5 * torch.log(-math.pi / y)
    
class GeneralizedNormal(TorchGeneralizedNormal, pyro.distributions.torch_distribution.TorchDistributionMixin):
    pass

import warnings
import pyro
import pyro.optim
import pyro.poutine as poutine
from pyro.infer.abstract_infer import TracePosterior
from pyro.infer.elbo import ELBO
from pyro.infer.util import torch_item

class ProblemSolveSVI(TracePosterior):
    """
    :param model: the model (callable containing Pyro primitives)
    :param guide: the guide (callable containing Pyro primitives)
    :param optim: a wrapper a for a PyTorch optimizer
    :type optim: ~pyro.optim.optim.PyroOptim
    :param loss: an instance of a subclass of :class:`~pyro.infer.elbo.ELBO`.
        Pyro provides three built-in losses:
        :class:`~pyro.infer.trace_elbo.Trace_ELBO`,
        :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO`, and
        :class:`~pyro.infer.traceenum_elbo.TraceEnum_ELBO`.
        See the :class:`~pyro.infer.elbo.ELBO` docs to learn how to implement
        a custom loss.
    :type loss: pyro.infer.elbo.ELBO
    :param num_samples: (DEPRECATED) the number of samples for Monte Carlo posterior approximation
    :param num_steps: (DEPRECATED) the number of optimization steps to take in ``run()``

    A unified interface for stochastic variational inference in Pyro. The most
    commonly used loss is ``loss=Trace_ELBO()``. See the tutorial
    `SVI Part I <http://pyro.ai/examples/svi_part_i.html>`_ for a discussion.
    """

    def __init__(
        self,
        model,
        guide,
        optim,
        loss,
        loss_and_grads=None,
        num_samples=0,
        num_steps=0,
        **kwargs
    ):
        if num_steps:
            warnings.warn(
                "The `num_steps` argument to SVI is deprecated and will be removed in "
                "a future release. Use `SVI.step` directly to control the "
                "number of iterations.",
                FutureWarning,
            )
        if num_samples:
            warnings.warn(
                "The `num_samples` argument to SVI is deprecated and will be removed in "
                "a future release. Use `pyro.infer.Predictive` class to draw "
                "samples from the posterior.",
                FutureWarning,
            )

        self.model = model
        self.guide = guide
        self.optim = optim
        self.num_steps = num_steps
        self.num_samples = num_samples
        super().__init__(**kwargs)

        if not isinstance(optim, pyro.optim.PyroOptim):
            raise ValueError(
                "Optimizer should be an instance of pyro.optim.PyroOptim class."
            )

        if isinstance(loss, ELBO):
            self.loss = loss.loss
            self.loss_and_grads = loss.loss_and_grads
        else:
            if loss_and_grads is None:

                def _loss_and_grads(*args, **kwargs):
                    loss_val = loss(*args, **kwargs)
                    if getattr(loss_val, "requires_grad", False):
                        loss_val.backward(retain_graph=True)
                    return loss_val

                loss_and_grads = _loss_and_grads
            self.loss = loss
            self.loss_and_grads = loss_and_grads

    def run(self, *args, **kwargs):
        """
        .. warning::
            This method is deprecated, and will be removed in a future release.
            For inference, use :meth:`step` directly, and for predictions,
            use the :class:`~pyro.infer.predictive.Predictive` class.
        """
        warnings.warn(
            "The `SVI.run` method is deprecated and will be removed in a "
            "future release. For inference, use `SVI.step` directly, "
            "and for predictions, use the `pyro.infer.Predictive` class.",
            FutureWarning,
        )
        if self.num_steps > 0:
            with poutine.block():
                for i in range(self.num_steps):
                    self.step(*args, **kwargs)
        return super().run(*args, **kwargs)


    def _traces(self, *args, **kwargs):
        for i in range(self.num_samples):
            guide_trace = poutine.trace(self.guide).get_trace(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, trace=guide_trace)
            ).get_trace(*args, **kwargs)
            yield model_trace, 1.0

    def evaluate_loss(self, *args, **kwargs):
            """
            :returns: estimate of the loss
            :rtype: float
    
            Evaluate the loss function. Any args or kwargs are passed to the model and guide.
            """
            with torch.no_grad():
                loss = self.loss(self.model, self.guide, *args, **kwargs)
                if isinstance(loss, tuple):
                    # Support losses that return a tuple, e.g. ReweightedWakeSleep.
                    return type(loss)(map(torch_item, loss))
                else:
                    return torch_item(loss)


    def step(self, *args, **kwargs):
            """
            :returns: estimate of the loss
            :rtype: float
    
            Take a gradient step on the loss function (and any auxiliary loss functions
            generated under the hood by `loss_and_grads`).
            Any args or kwargs are passed to the model and guide
            """
            # get loss and compute gradients
            with poutine.trace(param_only=True) as param_capture:
                loss = self.loss_and_grads(self.model, self.guide, *args, **kwargs)            
        
            if np.isnan(loss):
                with poutine.trace(param_only=True) as all_capture:
                    kwargs={'exploded':True}
                    self.loss_and_grads(self.model, self.guide, *args, **kwargs)
                [print(name,minmax(all_capture.trace.nodes[name]["value"])) for name in param_capture.trace.nodes.keys()]
                #[print(name,minmax(param_capture.trace.nodes[name]["value"].unconstrained())) for name in param_capture.trace.param_nodes]
                pyro.infer.util.zero_grads(params)
                return(param_capture)
            
            params = set(
                site["value"].unconstrained() for site in param_capture.trace.nodes.values()
            )
                
            # actually perform gradient steps
            # torch.optim objects gets instantiated for any params that haven't been seen yet
            self.optim(params)
    
            # zero gradients
            pyro.infer.util.zero_grads(params)
    
            if isinstance(loss, tuple):
                # Support losses that return a tuple, e.g. ReweightedWakeSleep.
                return type(loss)(map(torch_item, loss))
            else:
                return torch_item(loss)