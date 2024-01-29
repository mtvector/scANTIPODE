import model_distributions
from model_distributions import *
import model_functions
from model_functions import *
import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints
import contextlib
from contextlib import ExitStack, contextmanager

@contextmanager
def existing_plate_stack(plates, rightmost_dim=-1):
    """
    Create a contiguous stack of known :class:`plate` s with dimensions::

        rightmost_dim - len(sizes), ..., rightmost_dim

    :param str prefix: Name prefix for plates.
    :param iterable sizes: An iterable of plate sizes.
    :param int rightmost_dim: The rightmost dim, counting from the right.
    """
    assert rightmost_dim < 0
    with ExitStack() as stack:
        for plate_i in reversed(plates):
            if not isinstance(plate_i,contextlib.nullcontext):
                stack.enter_context(plate_i)
        yield


class MMB():
    '''
    MAP module base trivial class that holds the memory location of the overarching model
    '''
    def __init__(self,model):
        self.model=model


class MAPLaplaceModule(MMB):
    '''
    MAP module for a maximum a posteriori estimate given a laplacian prior. 
    Takes a list of plates or a list of nullcontext, where nullcontext represents a dependent dimension (from the right)
    '''
    def __init__(self,model,name,param_shape,plate_list=[],constraint=None,init_val=None,param_only=False):
        super().__init__(model)
        self.param_name=name
        self.param_shape=param_shape
        self.plate_list=plate_list
        self.dependent_dim=sum([isinstance(x,contextlib.nullcontext) for x in self.plate_list])
        self.constraint=constraint if constraint is not None else constraints.real
        self.init_val=init_val
        self.param_only=param_only
    
    def model_sample(self,s=torch.ones(1),scale=[]):
        if self.param_only:
            return self.make_params(s)
        with existing_plate_stack(scale+self.plate_list):
            return pyro.sample(self.param_name+'_sample',dist.Laplace(s.new_zeros(self.param_shape),
                            self.model.prior_scale*s.new_ones(self.param_shape)).to_event(self.dependent_dim))
    
    def make_params(self,s=torch.ones(1)):
        if self.init_val is not None:
            return pyro.param(self.param_name,self.init_val.to(s.device),constraint=self.constraint)
        else:
            return pyro.param(self.param_name,1e-5*torch.randn(self.param_shape,device=s.device),constraint=self.constraint)#1e-5*torch.randn

    def guide_sample(self,s=torch.ones(1),scale=[]):
        p=self.make_params(s)
        if self.param_only:
            return p
        with existing_plate_stack(scale+self.plate_list):
            return pyro.sample(self.param_name+'_sample',dist.Delta(p).to_event(self.dependent_dim))


class MAPHalfCauchyModule(MMB):
    def __init__(self,model,name,param_shape,plate_list=[],constraint=None,init_val=None,param_only=False):
        super().__init__(model)
        self.param_name=name
        self.param_shape=param_shape
        self.plate_list=plate_list
        self.dependent_dim=sum([isinstance(x,contextlib.nullcontext) for x in self.plate_list])
        self.constraint=constraint if constraint is not None else constraints.positive
        self.init_val=init_val
        self.param_only=param_only
    
    def model_sample(self,s=torch.ones(1),scale=[]):
        if self.param_only:
            return self.make_params(s)
        with existing_plate_stack(scale+self.plate_list):
            return pyro.sample(self.param_name+'_sample',dist.HalfCauchy(s.new_ones(self.param_shape)).to_event(self.dependent_dim))
    
    def make_params(self,s=torch.ones(1)):
        if self.init_val is not None:
            return pyro.param(self.param_name,self.init_val.to(s.device),constraint=self.constraint)
        else:
            return pyro.param(self.param_name,0.05*torch.ones(self.param_shape,device=s.device),constraint=self.constraint)#1e-5*torch.randn

    def guide_sample(self,s=torch.ones(1),scale=[]):
        p=self.make_params(s)
        if self.param_only:
            return p
        with existing_plate_stack(scale+self.plate_list):
            return pyro.sample(self.param_name+'_sample',dist.Delta(p))


class TreeEdges(MMB):
    def __init__(self, model,straight_through=True,zeros=True):
        super().__init__(model)
        self.straight_through=straight_through
        self.cat_dist= model_distributions.SafeAndRelaxedOneHotCategoricalStraightThrough if straight_through else model_distributions.SafeAndRelaxedOneHotCategorical
        if zeros:
            self.init_fn=torch.zeros
        else:
            self.init_fn=torch.randn
    
    def model_sample(self,s=torch.ones(1),approx=False):
        level_edges=self.make_params(s)
        if approx:
            temp=1.0
        else:
            temp=0.1
        level_edges=[pyro.sample('edges_sample_'+str(i),
                self.cat_dist(temperature=temp*torch.ones(1,device=s.device),logits=level_edges[i]).to_event(1))
                for i in range(len(self.model.level_sizes)-1)]
        return(level_edges)

    def make_params(self,s=torch.ones(1)):
        level_edges=[pyro.param('edges_'+str(i),
                0.01*self.init_fn(self.model.level_sizes[i+1],self.model.level_sizes[i]).to(s.device),
                                constraint=constraints.interval(-20,20)) 
                for i in range(len(self.model.level_sizes)-1)]
        return(level_edges)

    def guide_sample(self,s=torch.ones(1),approx=False):
        level_edges=self.make_params(s)
        if approx:
            temp=1.0
        else:
            temp=0.1
        level_edges=[pyro.sample('edges_sample_'+str(i),
                self.cat_dist(temperature=temp*torch.ones(1,device=s.device,requires_grad=False),logits=level_edges[i]).to_event(1))
                for i in range(len(self.model.level_sizes)-1)]
        return(level_edges)


class TreeConvergence(MMB):
    def __init__(self, model,strictness=1.):
        super().__init__(model)
        self.strictness=strictness

    def model_sample(self,y1,level_edges,s=torch.ones(1),strictness=None):
        if strictness is None:
            strictness=self.strictness
        #In the model sample up from the leaves of y1 but use ideal propagated values
        levels=[y1[...,self.model.level_indices[i]:self.model.level_indices[i+1]] for i in range(len(self.model.level_sizes))]
        result=levels[-1]
        results=[result]
        #Propagate from bottom to top
        for i in range(len(self.model.level_sizes) - 2, -1, -1):
            result=levels[i+1]@level_edges[i]
            results.append(result)
        results=results[::-1]
        with poutine.scale(scale=strictness):
            for i in range(len(self.model.level_sizes)):
                pyro.sample('y_level_'+str(i),dist.Laplace(results[i],s.new_ones(results[i].shape)).to_event(1))
        
            #Tree root prior is just a cost function (1 means no graph cycles,0 disconnected, >1 indicates cycles)
            pyro.sample('tree_root',dist.Laplace(s.new_ones(1,1),s.new_ones(1,1)).to_event(1))
        return(results)
        
    def guide_sample(self,y1,level_edges,s=torch.ones(1),strictness=None):
        if strictness is None:
            strictness=self.strictness
        #In the guide, sample up y1, no edge propagation
        levels=[y1[...,self.model.level_indices[i]:self.model.level_indices[i+1]] for i in range(len(self.model.level_sizes))]
        with poutine.scale(scale=strictness):
            for i in range(len(self.model.level_sizes)):
                pyro.sample('y_level_'+str(i),dist.Delta(levels[i]).to_event(1))

            #But still need to propagate edges to get the root value (check for cycles)
            results=[levels[-1]]
            for i in range(len(self.model.level_sizes) - 2, -1, -1):
                result=levels[i+1]@level_edges[i]
                results.append(result)
            results=results[::-1]
            pyro.sample('tree_root',dist.Delta(results[0]).to_event(1))
        return(results)


class TreeConvergenceBottomUp(MMB):
    def __init__(self, model,strictness=1.):
        super().__init__(model)
        self.strictness=strictness
        
    def model_sample(self,y1,level_edges,s=torch.ones(1),strictness=None):
        if strictness is None:
            strictness=self.strictness
        results=[y1]
        #Propagate from bottom to top
        for i in range(len(self.model.level_sizes) - 1):
            result=results[i]@level_edges[-(i+1)]
            results.append(result)
        results=results[::-1]
        
        #Tree root prior is just a cost function (1 means no graph cycles,0 disconnected, >1 indicates cycles)
        with poutine.scale(scale=strictness):
            pyro.sample('tree_root',dist.Laplace(s.new_ones(1,1),s.new_ones(1,1)).to_event(1))
        return(results)

    def guide_sample(self,y1,level_edges,s=torch.ones(1),strictness=None):
        if strictness is None:
            strictness=self.strictness
        results=[y1]
        for i in range(len(self.model.level_sizes) - 1):
            result=results[i]@level_edges[-(i+1)]
            results.append(result)
        results=results[::-1]
        with poutine.scale(scale=strictness):
            pyro.sample('tree_root',dist.Delta(results[0]).to_event(1))
        return(results)

    def just_propagate(self,y1,level_edges,s=torch.ones(1),strictness=None):
        results=[y1]
        for i in range(len(self.model.level_sizes) - 1):
            result=results[i]@level_edges[-(i+1)]
            results.append(result)
        results=results[::-1]
        return(results)

    def clean_propagate(self,y1,level_edges,s=torch.ones(1),strictness=None):
        results=[numpy_hardmax(y1)]
        for i in range(len(self.model.level_sizes) - 1):
            result=results[i]@level_edges[-(i+1)]
            results.append(numpy_hardmax(result))
        results=results[::-1]
        return(results)


class TreeConvergenceMaxBottomUp(MMB):
    def __init__(self, model,strictness=1.):
        super().__init__(model)
        self.strictness=strictness
        self.cat_dist=model_distributions.SafeAndRelaxedOneHotCategoricalStraightThrough
        
    def model_sample(self,y1,level_edges,s=torch.ones(1),strictness=None):
        if strictness is None:
            strictness=self.strictness
        results=[y1]
        #Propagate from bottom to top
        for i in range(len(self.model.level_sizes) - 1):
            result=results[i]@level_edges[-(i+1)]
            hard_dist = self.cat_dist(temperature = self.model.temperature * torch.ones(1,device=s.device), probs=result)
            out=pyro.sample('harden_level_'+str(i),hard_dist)
            print(result)
            print(out)
            results.append(out)
        results=results[::-1]
        
        return(results)

    def guide_sample(self,y1,level_edges,s=torch.ones(1),strictness=None):
        if strictness is None:
            strictness=self.strictness
        results=[y1]
        for i in range(len(self.model.level_sizes) - 1):
            result=results[i]@level_edges[-(i+1)]
            hard_dist = self.cat_dist(temperature = self.model.temperature * torch.ones(1,device=s.device), probs=result)
            out=pyro.sample('harden_level_'+str(i),hard_dist)
            results.append(out)
        results=results[::-1]
        return(results)

    def just_propagate(self,y1,level_edges,s=torch.ones(1),strictness=None):
        results=[numpy_hardmax(y1)]
        for i in range(len(self.model.level_sizes) - 1):
            result=results[i]@level_edges[-(i+1)]
            results.append(numpy_hardmax(result))
        results=results[::-1]
        return(results)