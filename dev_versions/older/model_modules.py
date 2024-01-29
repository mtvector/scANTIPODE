import model_distributions
from model_distributions import *
import model_functions
from model_functions import *

class SSC():
    '''
    Trivial class that simply holds the memory location of the overall model
    '''
    def __init__(self,model):
        self.model=model

class DM(SSC):
    '''Differential by module'''
    def __init__(self,model):
        super().__init__(model)
        
    def model_sample(self,s=torch.ones(1)):
        return pyro.sample('species_dm_sample',dist.Laplace(s.new_zeros(self.model.num_species,self.model.num_labels,self.model.num_latent),
                        self.model.prior_scale*s.new_ones(self.model.num_species,self.model.num_labels,self.model.num_latent)).to_event(3))
    
    def make_params(self,s=torch.ones(1)):
        return pyro.param('species_dm',s.new_zeros([self.model.num_species,self.model.num_labels,self.model.num_latent]))

    def guide_sample(self,s=torch.ones(1)):
        species_dm=self.make_params(s)
        return pyro.sample('species_dm_sample',dist.Delta(species_dm).to_event(3))

class BM(SSC):
    '''Batch effect by module'''
    def __init__(self,model):
        super().__init__(model)
        
    def model_sample(self,s=torch.ones(1)):
        return pyro.sample('batch_dm_sample',dist.Laplace(s.new_zeros(self.model.num_batch,self.model.num_labels,self.model.num_latent),
                        self.model.prior_scale*s.new_ones(self.model.num_batch,self.model.num_labels,self.model.num_latent)).to_event(3))
    
    def make_params(self,s=torch.ones(1)):
        return pyro.param('batch_dm',s.new_zeros([self.model.num_batch,self.model.num_labels,self.model.num_latent]))
    
    def guide_sample(self,s=torch.ones(1)):
        batch_dm=self.make_params(s)
        return pyro.sample('batch_dm_sample',dist.Delta(batch_dm).to_event(3))


class DI(SSC):
    '''Differential by identity'''
    def __init__(self, model):
        super().__init__(model)

    def model_sample(self,s=torch.ones(1)):
        return pyro.sample('species_di_sample', dist.Laplace(s.new_zeros(self.model.num_species, self.model.num_labels, self.model.num_var),
                        self.model.prior_scale*s.new_ones(self.model.num_species, self.model.num_labels, self.model.num_var)).to_event(3))

    def make_params(self,s=torch.ones(1)):
        return pyro.param('species_di', s.new_zeros([self.model.num_species, self.model.num_labels, self.model.num_var]))

    def guide_sample(self,s=torch.ones(1)):
        species_di=self.make_params(s)
        return pyro.sample('species_di_sample', dist.Delta(species_di).to_event(3))

class BInt(SSC):
    '''Batch intercept'''
    def __init__(self, model):
        super().__init__(model)

    def model_sample(self,s=torch.ones(1)):
        return pyro.sample('batch_intercept_sample', dist.Laplace(s.new_zeros(self.model.num_batch, self.model.num_var),
                        self.model.prior_scale*s.new_ones(self.model.num_batch, self.model.num_var)).to_event(2))

    def make_params(self,s=torch.ones(1)):
        return pyro.param('batch_intercept', s.new_zeros(self.model.num_batch, self.model.num_var))

    def guide_sample(self,s=torch.ones(1)):
        batch_intercept=self.make_params(s)
        return pyro.sample('batch_intercept_sample', dist.Delta(batch_intercept).to_event(2))

class CI(SSC):
    def __init__(self, model):
        super().__init__(model)

    def model_sample(self,s=torch.ones(1)):
        return pyro.sample('cluster_intercept_sample', dist.Laplace(s.new_zeros(self.model.num_labels, self.model.num_var),
                        self.model.prior_scale*s.new_ones(self.model.num_labels, self.model.num_var)).to_event(2))

    def make_params(self,s=torch.ones(1)):
        return pyro.param('cluster_intercept', s.new_zeros(self.model.num_labels, self.model.num_var))

    def guide_sample(self,s=torch.ones(1)):
        cluster_intercept=self.make_params(s)
        return pyro.sample('cluster_intercept_sample', dist.Delta(cluster_intercept).to_event(2))

class ZDW(SSC):
    def __init__(self, model):
        super().__init__(model)

    def model_sample(self,s=torch.ones(1)):
        return pyro.sample('z_decoder_weight_sample',dist.Laplace(s.new_zeros(self.model.num_latent,self.model.num_var), 
                        self.model.prior_scale*s.new_ones(self.model.num_latent,self.model.num_var)).to_event(2))

    def make_params(self,s=torch.ones(1)):
        return pyro.param('z_decoder_weight',(2/self.model.num_latent)*(torch.rand(self.model.num_latent,self.model.num_var).to(s.device)-0.5))

    def guide_sample(self,s=torch.ones(1)):
        z_decoder_weight=self.make_params(s)
        return pyro.sample('z_decoder_weight_sample',dist.Delta(z_decoder_weight).to_event(2))  

class DC(SSC):
    def __init__(self, model):
        super().__init__(model)
    
    def model_sample(self,s=torch.ones(1)):
        return pyro.sample('species_dc_sample',dist.Laplace(s.new_zeros(self.model.num_species,self.model.num_latent,self.model.num_var),
                        self.model.prior_scale*s.new_ones(self.model.num_species,self.model.num_latent,self.model.num_var)).to_event(3))                   

    def make_params(self,s=torch.ones(1)):
        return pyro.param('species_dc',(2/self.model.num_latent)*(torch.rand(self.model.num_species,self.model.num_latent,self.model.num_var).to(s.device)-0.5))

    def guide_sample(self,s=torch.ones(1)):
        dc=self.make_params(s)
        return pyro.sample('species_dc_sample',dist.Delta(dc).to_event(3))  

class BC(SSC):
    def __init__(self, model):
        super().__init__(model)  

    def model_sample(self,s=torch.ones(1)):
        #with poutine.scale(scale=self.model.num_batch/s.shape[0]):
        return pyro.sample('batch_dc_sample',dist.Laplace(s.new_zeros(s.shape[0],self.model.num_latent,self.model.num_var),
                        self.model.prior_scale*s.new_ones(s.shape[0],self.model.num_latent,self.model.num_var)).to_event(2))                   

    def make_params(self,bc_nn,batch,s=torch.ones(1)):
        return bc_nn(batch).reshape((s.shape[0],self.model.num_latent,self.model.num_var))

    def guide_sample(self,bc_nn,batch,s=torch.ones(1)):
        dc=self.make_params(bc_nn,batch,s)
        #with poutine.scale(scale=self.model.num_batch/s.shape[0]):
        return pyro.sample('batch_dc_sample',dist.Delta(dc).to_event(2))  


class BIFull(SSC):
    def __init__(self, model):
        super().__init__(model)  
    
    def model_sample(self,s=torch.ones(1)):
        #with poutine.scale(scale=self.model.num_batch/s.shape[0]):
        return pyro.sample('batch_di_full_sample',dist.Laplace(s.new_zeros(s.shape[0],self.model.num_var),
                        self.model.prior_scale*s.new_ones(s.shape[0],self.model.num_var)).to_event(1))                   

    def make_params(self,batch_nn,batch,y1,s=torch.ones(1)):
        return batch_nn(torch.cat([batch,y1],dim=-1))

    def guide_sample(self,batch_nn,batch,y1,s=torch.ones(1)):
        dc=self.make_params(batch_nn,batch,y1,s)
        #with poutine.scale(scale=self.model.num_batch/s.shape[0]):
        return pyro.sample('batch_di_full_sample',dist.Delta(dc).to_event(1))
    """Given infinite memory:
    def model_sample(self,s=torch.ones(1)):
        return pyro.sample('batch_di_sample', dist.Laplace(s.new_zeros(self.model.num_batch, self.model.num_labels, self.model.num_var),
                        self.model.prior_scale*s.new_ones(self.model.num_batch, self.model.num_labels, self.model.num_var)).to_event(3))

    def make_params(self,batch_nn,batch,y1,s=torch.ones(1)):
        return pyro.param('batch_di', s.new_zeros([self.model.num_batch, self.model.num_labels, self.model.num_var]))

    def guide_sample(self,batch_nn,batch,y1,s=torch.ones(1)):
        species_di=self.make_params(batch_nn,batch,y1,s)
        return pyro.sample('batch_di_sample', dist.Delta(species_di).to_event(3))
    """

class BIEmbed(SSC):
    def __init__(self, model):
        super().__init__(model)  
    
    def model_sample(self,s=torch.ones(1)):
        #with poutine.scale(scale=self.model.num_batch/self.model.batch_embed):?
        return pyro.sample('batch_di_sample',dist.Laplace(s.new_zeros(self.model.batch_embed,
                                                                      self.model.bi_depth_num,
                                                                      self.model.num_var),
                        self.model.prior_scale*s.new_ones(self.model.batch_embed,
                                                          self.model.bi_depth_num,
                                                          self.model.num_var)).to_event(3))                   

    def make_params(self,s=torch.ones(1)):
        return pyro.param('batch_di',s.new_zeros(self.model.batch_embed,
                                                 self.model.bi_depth_num,
                                                 self.model.num_var))

    def guide_sample(self,s=torch.ones(1)):
        bi=self.make_params(s)
        #with poutine.scale(scale=self.model.num_batch/self.model.batch_embed):?
        return pyro.sample('batch_di_sample',dist.Delta(bi).to_event(3))

class ZLoc(SSC):
    def __init__(self, model):
        super().__init__(model)

    def model_sample(self,s=torch.ones(1)):
        return(pyro.sample("locs_sample", dist.Laplace(
        s.new_zeros((self.model.num_labels,self.model.num_latent)),
        self.model.prior_scale*s.new_ones((self.model.num_labels,self.model.num_latent))
        ).to_event(2)))

    def make_params(self,s=torch.ones(1)):
        return pyro.param("locs",
                          s.new_zeros((self.model.num_labels,self.model.num_latent)))

    def guide_sample(self,s=torch.ones(1)):
        locs=self.make_params(s)
        return pyro.sample('locs_sample',dist.Delta(locs).to_event(2))


class ZScale(SSC):
    def __init__(self, model):
        super().__init__(model)

    def model_sample(self,s=torch.ones(1)):
        return(pyro.sample("scales_sample", dist.HalfCauchy(
            s.new_ones((self.model.num_labels,self.model.num_latent))
        ).to_event(2)))

    def make_params(self,s=torch.ones(1)):
        return pyro.param("scales",
                          0.1*s.new_ones((self.model.num_labels,self.model.num_latent)),
                          constraint=constraints.positive)

    def guide_sample(self,s=torch.ones(1)):
        scales=self.make_params(s)
        return pyro.sample('scales_sample',dist.Delta(scales).to_event(2))

class ZLocDynam(SSC):
    def __init__(self, model):
        super().__init__(model)

    def model_sample(self,s=torch.ones(1)):
        return(pyro.sample("locs_dynam_sample", dist.Laplace(
        s.new_zeros((self.model.num_labels,self.model.num_latent)),
        self.model.prior_scale*s.new_ones((self.model.num_labels,self.model.num_latent))
        ).to_event(2)))

    def make_params(self,s=torch.ones(1)):
        return pyro.param("locs_dynam",
                          s.new_zeros((self.model.num_labels,self.model.num_latent)))

    def guide_sample(self,s=torch.ones(1)):
        locs=self.make_params(s)
        return pyro.sample('locs_dynam_sample',dist.Delta(locs).to_event(2))


class TreeEdges(SSC):
    def __init__(self, model,straight_through=True):
        super().__init__(model)
        self.straight_through=straight_through
        self.cat_dist= model_distributions.SafeAndRelaxedOneHotCategoricalStraightThrough if straight_through else model_distributions.SafeAndRelaxedOneHotCategorical
        
    def model_sample(self,s=torch.ones(1),approx=False):
        if approx:
            temp=1.0
        else:
            temp=0.1
        level_edges=[pyro.sample('edges_sample_'+str(i),
                self.cat_dist(temperature=temp*s.new_ones(1),
                              logits=s.new_zeros(self.model.level_sizes[i+1],self.model.level_sizes[i])).to_event(1))
                for i in range(len(self.model.level_sizes)-1)]
        return(level_edges)

    def make_params(self,s=torch.ones(1)):
        level_edges=[pyro.param('edges_'+str(i),
                0.01*torch.randn(self.model.level_sizes[i+1],self.model.level_sizes[i]).to(s.device),
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
                self.cat_dist(temperature=temp*s.new_ones(1),logits=level_edges[i]).to_event(1))
                for i in range(len(self.model.level_sizes)-1)]
        return(level_edges)


class TreeConvergence(SSC):
    def __init__(self, model):
        super().__init__(model)

    def model_sample(self,y1,level_edges,s=torch.ones(1)):
        #In the model sample up from the leaves of y1 but use ideal propagated values
        levels=[y1[...,self.model.level_indices[i]:self.model.level_indices[i+1]] for i in range(len(self.model.level_sizes))]
        result=levels[-1]
        results=[result]
        #Propagate from bottom to top
        for i in range(len(self.model.level_sizes) - 2, -1, -1):
            result=levels[i+1]@level_edges[i]
            results.append(result)
        results=results[::-1]

        for i in range(len(self.model.level_sizes)):
            pyro.sample('y_level_'+str(i),dist.Laplace(results[i],s.new_ones(results[i].shape)).to_event(1))
        
        #Tree root prior is just a cost function (1 means no graph cycles,0 disconnected, >1 indicates cycles)
        pyro.sample('tree_root',dist.Laplace(s.new_ones(1,1),s.new_ones(1,1)).to_event(1))
        return(results)
        
    def guide_sample(self,y1,level_edges,s=torch.ones(1)):
        #In the guide, sample up y1, no edge propagation
        levels=[y1[...,self.model.level_indices[i]:self.model.level_indices[i+1]] for i in range(len(self.model.level_sizes))]
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