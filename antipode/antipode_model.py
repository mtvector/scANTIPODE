#Derived from version PBS1.9.0
import os
import sklearn
from sklearn import cluster
import pandas as pd
import scanpy as sc
import scvi
import inspect
import tqdm
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn.functional import softplus, softmax
from torch.distributions import constraints
import seaborn
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.optim
from pyro.infer import SVI
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
from scvi.module.base import PyroBaseModuleClass

class ANTIPODE(PyroBaseModuleClass,AntipodeTrainingMixin):#
    '''
    ANTIPODE (Single Cell Ancestral Node Taxonomy Inference by Parcellation of Differential Expression) 
    is a variational inference model developed for the simultaneous analysis (DE) and 
    categorization (taxonomy generation) of cell types across evolution (or now any covariate) using single-cell RNA-seq data.
    

    Parameters:
    adata (AnnData): An AnnData object containing the single-cell dataset.
    discov_pair (tuple): A tuple indicating the key and location of the discovery covariate 
                         in the AnnData object. Format: ('key', 'location'), where location is 
                         either 'obs' or 'obsm'.
    batch_pair (tuple): A tuple indicating the key and location of the batch covariate 
                        in the AnnData object. Format: ('key', 'location'), where location is 
                        either 'obs' or 'obsm'.
    num_var (int): Number of variables (features) in the dataset.
    level_sizes (list of int, optional): Sizes of each level in the model's hierarchical structure. 
                                         Defaults to [1, 10, 25, 50].
    num_latent (int, optional): Number of latent dimensions. Defaults to 50.
    scale_factor (float, optional): Scaling factor for data normalization. If None, it is inferred from the data.
    prior_scale (float, optional): Scale of the Laplace prior distributions. Defaults to 100.
    dcd_prior (float, optional): Scale of the prior for the decoder. If None, defaults to a specific inferred value.
    decay_function (callable, optional): A function that defines the decay of certain parameters over iterations.
    max_strictness (float, optional): Maximum strictness parameter for tree convergence. Defaults to 1.
    bi_depth (int, optional): Depth of the tree for the approximation of batch by identity effects. Defaults to 2.
    num_batch_embed (int, optional): Number of batch embeddings. Defaults to 10.
    classifier_hidden (list of int, optional): Sizes of hidden layers for the classifier network. Defaults to [3000, 3000, 3000].
    encoder_hidden (list of int, optional): Sizes of hidden layers for the encoder network. Defaults to [6000, 5000, 3000, 1000].
    '''
    def __init__(self, adata, discov_pair, batch_pair, layer, num_var, level_sizes=[1, 10, 25, 50], 
                 num_latent=50, scale_factor=None, prior_scale=100,dcd_prior=None,
                 decay_function=None, max_strictness=1., bi_depth=2, num_batch_embed=10,
                 classifier_hidden=[3000,3000,3000],encoder_hidden=[6000,5000,3000,1000]):
        
        # Determine num_discov and num_batch from the AnnData object
        self.discov_loc, self.discov_key = discov_pair
        self.batch_loc, self.batch_key = batch_pair
        self.num_discov = adata.obsm[self.discov_key].shape[-1] if self.discov_loc == 'obsm' else len(adata.obs[self.discov_key].unique())
        self.num_batch = adata.obsm[self.batch_key].shape[-1] if self.batch_loc == 'obsm' else len(adata.obs[self.batch_key].unique())
        self.design_matrix = (self.discov_loc == 'obsm')
        self.layer=layer

        self._setup_adata_manager_store: dict[str, type[scvi.data.AnnDataManager]] = {}
        self.num_var = adata.layers[layer].shape[-1]
        self.num_latent = num_latent
        self.scale_factor = scale_factor if scale_factor is not None else 2e2 / (num_var * num_labels * num_latent)
        self.level_sizes = level_sizes
        self.num_labels = np.sum(self.level_sizes)
        self.level_indices = np.cumsum([0] + self.level_sizes)
        self.bi_depth = bi_depth
        self.num_bi_depth = sum(self.level_sizes[:self.bi_depth])
        self.num_batch_embed = num_batch_embed
        self.max_strictness = 1.
        self.decay_function = gen_linear_function(max_steps,10000) if decay_function is None else decay_function 
        self.temperature = 0.1
        self.epsilon = 0.006
        self.approx = False
        self.prior_scale = prior_scale
        
        self.dcd_prior=torch.zeros((self.num_discov,self.num_var)) if dcd_prior is None else dcd_prior#Use this for 
        
        # Initialize plates to be used during sampling
        self.var_plate=pyro.plate('var_plate',self.num_var,dim=-1)
        self.discov_plate=pyro.plate('discov_plate',self.num_discov,dim=-3)
        self.batch_plate=pyro.plate('batch_plate',self.num_batch,dim=-3)
        self.latent_plate=pyro.plate('latent_plate',self.num_latent,dim=-1)
        self.latent_plate2=pyro.plate('latent_plate2',self.num_latent,dim=-2)
        self.label_plate=pyro.plate('label_plate',self.num_labels,dim=-2)
        self.batch_embed_plate=pyro.plate('batch_embed_plate',self.num_batch_embed,dim=-3)
        self.bi_depth_plate=pyro.plate('bi_depth_plate',self.num_bi_depth,dim=-2)

        #Initialize MAP inference modules
        self.dm=MAPLaplaceModule(self,'discov_dm',[self.num_discov,self.num_labels,self.num_latent],[self.discov_plate,self.label_plate,self.latent_plate])
        self.bm=MAPLaplaceModule(self,'batch_dm',[self.num_batch,self.num_labels,self.num_latent],[self.batch_plate,self.label_plate,self.latent_plate])
        self.di=MAPLaplaceModule(self,'discov_di',[self.num_discov,self.num_labels,self.num_var],[self.discov_plate,self.label_plate,self.var_plate])
        self.bei=MAPLaplaceModule(self,'batch_di',[self.num_batch_embed,self.num_bi_depth,self.num_var],[self.batch_embed_plate,self.bi_depth_plate,self.var_plate])
        self.ci=MAPLaplaceModule(self,'cluster_intercept',[self.num_labels, self.num_var],[self.label_plate,self.var_plate])
        self.dc=MAPLaplaceModule(self,'discov_dc',[self.num_discov,self.num_latent,self.num_var],[self.discov_plate,self.latent_plate2,self.var_plate])
        self.zdw=MAPLaplaceModule(self,'z_decoder_weight',[self.num_latent,self.num_var],[self.latent_plate2,self.var_plate],init_val=((2/self.num_latent)*(torch.rand(self.num_latent,self.num_var)-0.5)),param_only=False)
        self.zl=MAPLaplaceModule(self,'locs',[self.num_labels,self.num_latent],[self.label_plate,self.latent_plate],param_only=False)
        self.zs=MAPLaplaceModule(self,'scales',[self.num_labels,self.num_latent],[self.label_plate,self.latent_plate],init_val=0.01*torch.ones(self.num_labels,self.num_latent),constraint=constraints.positive,param_only=False)
        self.zld=MAPLaplaceModule(self,'locs_dynam',[self.num_labels,self.num_latent],[self.label_plate,self.latent_plate],param_only=False)
        
        self.tree_edges=TreeEdges(self,straight_through=True)
        self.tree_convergence=TreeConvergence(self,strictness=1.)        
        self.tree_convergence_bottom_up=TreeConvergenceBottomUp(self,strictness=1.)        
        self.z_transform=null_function#centered_sigmoid#torch.special.expit

        if self.design_matrix:
            fields={'s':('layers','spliced'),
            'discov_ind':('obsm','discov_onehot'),
            'batch_ind':('obsm','batch_onehot')}
            field_types={"s":np.float32,"batch_ind":np.float32,"discov_ind":np.float32}
        else:
            fields={'s':('layers','spliced'),
            'discov_ind':('obs','species'),
            'batch_ind':('obs','batch_name')}
            field_types={"s":np.float32,"batch_ind":np.int64,"discov_ind":np.int64}

        self.fields=fields
        self.field_types=field_types
        self.setup_anndata(adata, {'discov_ind': discov_pair, 'batch_ind': batch_pair}, self.field_types)
        
        super().__init__()
        # Setup the various neural networks used in the model and guide
        self.z_decoder=ZDecoder(num_latent=self.num_latent, num_var=num_var, hidden_dims=[])        
        self.zl_encoder=ZLEncoder(num_var=num_var,hidden_dims=encoder_hidden,num_cat_input=self.num_discov,
                    outputs=[(self.num_latent,None),(self.num_latent,softplus)])
        
        self.classifier=Classifier(num_latent=self.num_latent,hidden_dims=classifier_hidden,
                    outputs=[(self.num_labels,None),(1,None),(1,softplus)])

        #self.bc_nn=SimpleFFNN(in_dim=self.num_batch,hidden_dims=[200,200,50,5],
        #            out_dim=self.num_var*self.num_latent)
        #Too large to exactly model gene-level batch effects for all cluster x batch
        self.be_nn=SimpleFFNN(in_dim=self.num_batch,hidden_dims=[1000,500,500],
                    out_dim=self.num_batch_embed)
        
        self.epsilon = 0.006
        #Initialize model in approximation mode
        self.approx=False
        self.prior_scale=prior_scale
        self.args=inspect.getfullargspec(self.model).args[1:]#skip self

    def setup_anndata(self,adata: anndata.AnnData,fields,field_types,**kwargs,):
        
        anndata_fields=[make_field(x,self.fields[x]) for x in self.fields.keys()]
        
        adata_manager = scvi.data.AnnDataManager(
            fields=anndata_fields
        )
        adata_manager.register_fields(adata, **kwargs)
        self.register_manager(adata_manager)
        if fields['discov_ind'][0]=='obsm':
            self.design_matrix=True
            if fields['batch_ind'][0]!='obsm':
                raise Exception("If discov is design matrix, batch must be as well!")


    def register_manager(self, adata_manager: scvi.data.AnnDataManager):
        adata_id = adata_manager.adata_uuid
        self._setup_adata_manager_store[adata_id] = adata_manager
        self.adata_manager=adata_manager
    
    def set_approx(self,b: bool):
        self.approx=b
        
    def model(self, s,discov_ind=torch.zeros(1),batch_ind=torch.zeros(1),step=torch.ones(1),taxon=torch.zeros(1)):
        # Register various nn.Modules (i.e. the decoder/encoder networks) with Pyro
        pyro.module("antipode", self)

        if not self.design_matrix:
            batch=index_to_onehot(batch_ind,[s.shape[0],self.num_batch]).to(s.device)
            discov=index_to_onehot(discov_ind,[s.shape[0],self.num_discov]).to(s.device)
            batch_ind=batch_ind.squeeze()
            discov_ind=discov_ind.squeeze()
        else:
            batch=batch_ind
            discov=discov_ind
        
        minibatch_plate=pyro.plate("minibatch_plate", s.shape[0],dim=-1)
        minibatch_plate2=pyro.plate("minibatch_plate2", s.shape[0],dim=-2)
        cur_strictness=self.decay_function(step, self.max_strictness)
        l = s.sum(1).unsqueeze(-1)
        
        # We scale all sample statements by scale_factor so that the ELBO loss function
        # is normalized wrt the number of datapoints and genes.
        # This helps with numerical stability during optimization.
        with poutine.scale(scale=self.scale_factor):
            # This gene-level parameter modulates the variance of the observation distribution
            s_theta = pyro.param("s_inverse_dispersion", 50.0 * s.new_ones(self.num_var),
                               constraint=constraints.positive)
            
            dcd=pyro.param("discov_constitutive_de", self.dcd_prior.to(s.device))
            level_edges=self.tree_edges.model_sample(s,approx=self.approx)
            
            with minibatch_plate:
                beta_prior_a=1.*s.new_ones(self.num_labels)
                beta_prior_a[0]=10.
                if self.approx:#Bernoulli particles approx?
                    taxon_probs = pyro.sample("taxon_probs", dist.Beta(beta_prior_a,s.new_ones(self.num_labels),validate_args=True).to_event(1))
                    taxon = pyro.sample('taxon',dist.RelaxedBernoulli(temperature=0.1*s.new_ones(1),probs=taxon_probs).to_event(1))
                    #self.tree_convergence.model_sample(taxon,level_edges,s,cur_strictness)#Someday will be possible to properly generate Undirected acyclic graphs here
                else:
                    taxon_probs=pyro.sample('taxon_probs',dist.Dirichlet(s.new_ones(s.shape[0],self.level_sizes[-1]),validate_args=True))
                    if sum(taxon.shape) > 1:#Supervised?
                        if taxon.shape[-1]==self.num_labels:#Totally supervised?
                            pass
                        else:#Only bottom layer is supervised?
                            taxon = pyro.sample("taxon", dist.OneHotCategorical(probs=taxon_probs,validate_args=True),obs=taxon)
                            taxon = torch.concat(self.tree_convergence_bottom_up.just_propagate(taxon,level_edges,s),-1)
                    else:#Unsupervised
                        taxon = pyro.sample("taxon", dist.OneHotCategorical(probs=taxon_probs,validate_args=True),infer={'enumerate':'parallel'})
                        taxon = torch.concat(self.tree_convergence_bottom_up.just_propagate(taxon,level_edges,s),-1)
                    taxon_probs=self.tree_convergence_bottom_up.just_propagate(taxon_probs,level_edges,s,cur_strictness)
                    taxon_probs=torch.cat(taxon_probs,-1)
                   
            locs=self.zl.model_sample(s)
            scales=self.zs.model_sample(s)
            locs_dynam=self.zld.model_sample(s)
            discov_dm=self.dm.model_sample(s,scale=fest([discov,taxon_probs],-1))
            discov_di=self.di.model_sample(s,scale=fest([discov,taxon_probs],-1))
            batch_dm=self.bm.model_sample(s,scale=fest([batch,taxon_probs],-1))
            batch_embed=torch.sigmoid(self.be_nn(batch))
            bei=self.bei.model_sample(s,scale=fest([batch_embed,taxon_probs[...,:self.num_bi_depth]],-1))
            cluster_intercept=self.ci.model_sample(s,scale=fest([taxon_probs],-1))
            z_decoder_weight=self.zdw.model_sample(s)
            
            with minibatch_plate:
                bi=torch.einsum('...bi,...ijk->...bjk',batch_embed,bei)
                bi=torch.einsum('...bj,...bjk->...bk',taxon[...,:self.num_bi_depth],bi)
                psi = centered_sigmoid(pyro.sample('psi',dist.Laplace(s.new_zeros(s.shape[0],1),self.prior_scale*s.new_ones(s.shape[0],1)).to_event(1)))#Used to be normal#maybe should be centered sigmoid
                this_locs=oh_index(locs,taxon)
                this_scales=oh_index(scales,taxon)
                z=pyro.sample('z', dist.Normal(this_locs,this_scales+self.epsilon,validate_args=True).to_event(1))

            cur_discov_dm = oh_index1(discov_dm, discov_ind) if self.design_matrix else discov_dm[discov_ind]
            cur_batch_dm = oh_index1(batch_dm, batch_ind) if self.design_matrix else batch_dm[batch_ind]
            cur_dcd = oh_index(dcd, discov) if self.design_matrix else  dcd[discov_ind]
            
            z=z+oh_index2(cur_discov_dm,taxon) + oh_index2(cur_batch_dm,taxon)+(oh_index(locs_dynam,taxon)*psi)
            z=self.z_transform(z)                
            fake_z=oh_index(locs,taxon_probs)+oh_index2(discov_dm[discov_ind],taxon_probs) + oh_index2(batch_dm[batch_ind],taxon_probs)+(oh_index(locs_dynam,taxon_probs)*psi)
            fake_z=self.z_transform(fake_z)
            discov_dc=self.dc.model_sample(s,scale=fest([discov,fake_z.abs()],-1))
            cur_discov_di = oh_index1(discov_di, discov_ind) if self.design_matrix else discov_di[discov_ind]
            cur_discov_dc = oh_index1(discov_dc, discov_ind) if self.design_matrix else discov_dc[discov_ind]
            cur_discov_di=oh_index2(cur_discov_di,taxon)
            cur_cluster_intercept=oh_index(cluster_intercept,taxon)
            
            mu=torch.einsum('...bi,...bij->...bj',z,z_decoder_weight+cur_discov_dc)#+bc
            spliced_mu=mu+cur_dcd+cur_discov_di+cur_cluster_intercept+bi
            spliced_out=torch.softmax(spliced_mu,dim=-1)
            log_mu = (l * spliced_out + 1e-6).log()
            
            with self.var_plate,minibatch_plate2:
                s_dist = dist.NegativeBinomial(total_count=s_theta,logits=log_mu-s_theta.log(),validate_args=True)
                s_out=pyro.sample("s", s_dist, obs=s.int())

    
    # The guide specifies the variational distribution
    def guide(self, s,discov_ind=torch.zeros(1),batch_ind=torch.zeros(1),step=torch.ones(1),taxon=torch.zeros(1)):
        pyro.module("antipode", self)
        
        if not self.design_matrix:
            batch=index_to_onehot(batch_ind,[s.shape[0],self.num_batch]).to(s.device)
            discov=index_to_onehot(discov_ind,[s.shape[0],self.num_discov]).to(s.device)
            batch_ind=batch_ind.squeeze()
            discov_ind=discov_ind.squeeze()
        else:
            batch=batch_ind
            discov=discov_ind
        
        minibatch_plate=pyro.plate("minibatch_plate", s.shape[0])
        cur_strictness=self.decay_function(step, self.max_strictness)
        
        with poutine.scale(scale=self.scale_factor):
            locs=self.zl.guide_sample(s)
            scales=self.zs.guide_sample(s)
            locs_dynam=self.zld.guide_sample(s)
            level_edges=self.tree_edges.guide_sample(s,approx=self.approx) 
            with minibatch_plate:
                z_loc, z_scale= self.zl_encoder(s,discov)
                z=pyro.sample('z',dist.Normal(z_loc,z_scale+self.epsilon).to_event(1))
                z=self.z_transform(z)
                taxon_logits,psi_loc,psi_scale=self.classifier(z)
                psi=centered_sigmoid(pyro.sample('psi',dist.Normal(psi_loc,psi_scale).to_event(1)))
                if self.approx:
                    taxon_dist = dist.Delta(safe_sigmoid(taxon_logits),validate_args=True).to_event(1)
                    taxon_probs = pyro.sample("taxon_probs", taxon_dist)
                    taxon = pyro.sample('taxon',dist.RelaxedBernoulli(temperature=self.temperature*s.new_ones(1),probs=taxon_probs).to_event(1))
                    #self.tree_convergence.guide_sample(taxon,level_edges,s,cur_strictness)
                else:
                    taxon_probs=pyro.sample('taxon_probs',dist.Delta(safe_softmax(taxon_logits[...,-self.level_sizes[-1]:])).to_event(1))
                    if sum(taxon.shape) > 1:
                        pass
                    else:
                        taxon = pyro.sample("taxon", 
                                         dist.OneHotCategorical(probs=taxon_probs,validate_args=True),infer={'enumerate':'parallel'})                    
                    if taxon.shape[-1]<self.num_labels:
                        taxon = torch.concat(self.tree_convergence_bottom_up.just_propagate(taxon,level_edges,s),-1)

                    taxon_probs=self.tree_convergence_bottom_up.just_propagate(taxon_probs[...,-self.level_sizes[-1]:],level_edges,s,cur_strictness)
                    taxon_probs=torch.cat(taxon_probs,-1)
                
            z_decoder_weight=self.zdw.guide_sample(s)
            discov_dm=self.dm.guide_sample(s,scale=fest([discov,taxon_probs],-1))
            batch_dm=self.bm.guide_sample(s,scale=fest([batch,taxon_probs],-1))
            batch_embed=torch.sigmoid(self.be_nn(batch))
            discov_di=self.di.guide_sample(s,scale=fest([discov,taxon_probs],-1))
            cluster_intercept=self.ci.guide_sample(s,scale=fest([taxon_probs],-1))
            bei=self.bei.guide_sample(s,scale=fest([batch_embed,taxon_probs[...,:self.num_bi_depth]],-1))#maybe should be abs sum bei
            if self.design_matrix:
                z=z+oh_index2(oh_index1(discov_dm,discov_ind),taxon) + oh_index2(oh_index1(batch_dm,batch_ind),taxon)+(oh_index(locs_dynam,taxon)*psi)
            else:
                z=z+oh_index2(discov_dm[discov_ind],taxon) + oh_index2(batch_dm[batch_ind],taxon)+(oh_index(locs_dynam,taxon)*psi)
            z=self.z_transform(z)
            fake_z=oh_index(locs,taxon_probs)+oh_index2(discov_dm[discov_ind],taxon_probs) + oh_index2(batch_dm[batch_ind],taxon_probs)+(oh_index(locs_dynam,taxon_probs)*psi)
            fake_z=self.z_transform(fake_z)
            discov_dc=self.dc.guide_sample(s,scale=fest([discov,fake_z.abs()],-1))

class AntipodeTrainingMixin:
    '''
    Mixin class providing functions to actually run ANTIPODE
    can use supervised taxonomy by training only phase2
    '''
    
    def save_params_to_uns(self,prefix=''):
        pstore=param_store_to_numpy()
        pstore={n:pstore[n] for n in pstore.keys() if not re.search('encoder|classifier',n)}
        self.adata_manager.adata.uns[prefix+'param_store']=pstore

    def store_outputs(self,device='cuda',prefix=''):
        self.save_params_to_uns(prefix='')
        self.to('cpu')
        self.eval()
        antipode_outs=get_antipode_outputs(self,batch_size=2048,device=device)
        taxon=antipode_outs[1][0]
        adata.obsm['X_antipode']=antipode_outs[0][0]
        level_edges=[numpy_hardmax(self.adata_manager.adata.uns['param_store']['edges_'+str(i)],axis=-1) for i in range(len(self.level_sizes)-1)]
        levels=self.tree_convergence_bottom_up.just_propagate(scipy.special.softmax(taxon[...,-self.level_sizes[-1]:],axis=-1),level_edges,s=torch.ones(1))
        prop_taxon=np.concatenate(levels,axis=-1)
        adata.obsm['taxon_probs']=prop_taxon
        levels=self.tree_convergence_bottom_up.just_propagate(numpy_hardmax(levels[-1],axis=-1),level_edges,s=torch.ones(1))
        for i in range(len(levels)):
            cur_clust=prefix+'level_'+str(i)
            self.adata_manager.adata.obs[cur_clust]=levels[i].argmax(1)
            self.adata_manager.adata.obs[cur_clust]=adata.obs[cur_clust].astype(str)
        self.adata_manager.adata.obs[prefix+'antipode_cluster'] = self.adata_manager.adata.obs.apply(lambda x: '_'.join([x[prefix+'level_'+str(i)] for i in range(len(levels))]), axis=1)
    
    def train_phase_1(self,max_steps,print_every=10000,device='cuda',max_learning_rate=0.001,num_particles=3,one_cycle_lr=True,steps=0,batch_size=32):
        #particle phase
        steps=steps
        print(self.fields)
        print(self.field_types)
        dataloader=scvi.dataloaders.AnnDataLoader(self.adata_manager,batch_size=32,drop_last=True,shuffle=True,data_and_attributes=self.field_types)#supervised_field_types for supervised step
        scheduler=pyro.optim.OneCycleLR({'max_lr':max_learning_rate,'total_steps':max_steps,'div_factor':100,'optim_args':{},'optimizer':torch.optim.Adam}) if one_cycle_lr else pyro.optim.ClippedAdam({'lr':max_learning_rate,'lrd':(1-(5e-6))})
        elbo = pyro.infer.JitTrace_ELBO(num_particles=num_particles,strict_enumeration_warning=False)
        svi = SVI(self.model, self.guide, scheduler, elbo)
        self.train()
        self.zl_encoder.train()
        
        self=self.to(device)
        self.set_approx(True)
        loss_tracker=[]
        pbar = tqdm.tqdm(total=max_steps, position=0)
        done=False
        while steps < max_steps:
            for x in dataloader:
                x['step']=torch.ones(1).to(device)*steps
                x=[x[k].to(device) if k in x.keys() else torch.zeros(1) for k in self.args]
                loss=svi.step(*x)
                steps+=1
                if steps<max_steps-1:
                    scheduler.step()
                else:
                    break
                pbar.update(1)
                loss_tracker.append(loss)
                if steps%print_every == 0:
                    # Tell the scheduler we've done one epoch.
                    pbar.write("[Step %02d]  Loss: %.5f" % (steps, np.mean(loss_tracker[-print_every:])))
        
        pbar.close()
        allDone()
        print("Finished training!")

    def prepare_phase_2(self):
        '''Run this if not running in supervised only mode (JUST phase2 with provided obsm clustering), runs kmeans, resets tree edges and locs'''
        kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=self.level_sizes[-1],init='k-means++',max_iter=1000,reassignment_ratio=0.001,n_init=100,random_state=0).fit(self.adata_manager.adata.obsm['X_antipode'])
        adata.obs['kmeans']=kmeans.labels_
        adata.obs['kmeans']=adata.obs['kmeans'].astype(str).astype('category')
        adata.obsm['kmeans_onehot']=numpy_onehot(adata.obs['kmeans'].cat.codes,num_classes=self.level_sizes[-1]) #yoh=yoh+1e-10;yoh=oh/oh.sum(-1).reshape(-1,1)#for relaxed
        new_locs=torch.concatenate(
            [pyro.param('locs').new_zeros(sum(self.level_sizes[:-1]),pyro.param('locs').shape[1]),
             torch.tensor(kmeans.cluster_centers_,device=pyro.param('locs').device)],
             axis=0)
        pyro.get_param_store().__setitem__('locs',new_locs)
        pyro.get_param_store().__setitem__('locs_dynam',new_locs.new_zeros(new_locs.shape))
        for n in pyro.get_param_store():
            if 'edge' in n:
                pyro.get_param_store().__setitem__(n,pyro.param(n).new_zeros(pyro.param(n).shape))
        
    def train_phase_2(self,max_steps,taxon_label='kmeans_onehot',print_every=10000,device='cuda',max_learning_rate=0.001,num_particles=1,one_cycle_lr=False,steps=0,batch_size=32):
        '''empirically works best and fastest with one_cycle_lr=False'''
        steps=steps
        supervised_field_types=self.field_types.copy()
        supervised_fields=self.fields.copy()
        supervised_field_types["taxon"]=np.float32
        self.adata_manager.register_new_fields([make_field('taxon',('obsm',taxon_label))])
        class_dataloader=scvi.dataloaders.AnnDataLoader(self.adata_manager,batch_size=batch_size,drop_last=True,shuffle=True,data_and_attributes=supervised_field_types)
        scheduler=pyro.optim.OneCycleLR({'max_lr':max_learning_rate,'total_steps':max_steps,'div_factor':100,'optim_args':{},'optimizer':torch.optim.Adam}) if one_cycle_lr else pyro.optim.ClippedAdam({'lr':max_learning_rate,'lrd':(1-(5e-6))})
        elbo = pyro.infer.JitTrace_ELBO(num_particles=num_particles,strict_enumeration_warning=False)
        svi = SVI(self.model, self.guide, scheduler, elbo)
        
        self.train()
        self=self.to(device)
        self.set_approx(False)
        loss_tracker=[]
        #for steps in range(max_steps):
        pbar = tqdm.tqdm(total=max_steps, position=0)
        done=False
        while steps < max_steps:
            for x in class_dataloader:
                x['step']=torch.ones(1).to(device)*steps
                x=[x[k].to(device) if k in x.keys() else torch.zeros(1) for k in self.args]
                loss=svi.step(*x)
                steps+=1
                if steps<=max_steps-1:
                    #scheduler.step()
                    pass
                else:
                    break
                pbar.update(1)
                loss_tracker.append(loss)
                if steps%print_every == 0:
                    # Tell the scheduler we've done one epoch.
                    pbar.write("[Step %02d]  Loss: %.5f" % (steps, np.mean(loss_tracker[-print_every:])))
        
        pbar.close()
        allDone()
        print("Finished training!")

    def train_phase_3(self,max_steps,print_every=10000,device='cuda',max_learning_rate=0.001,num_particles=3,one_cycle_lr=True,steps=0,batch_size=32):
        steps=steps
        dataloader=scvi.dataloaders.AnnDataLoader(self.adata_manager,batch_size=batch_size,drop_last=True,shuffle=True,data_and_attributes=self.field_types)#supervised_field_types for supervised step
        scheduler=pyro.optim.OneCycleLR({'max_lr':max_learning_rate,'total_steps':max_steps,'div_factor':100,'optim_args':{},'optimizer':torch.optim.Adam}) if one_cycle_lr else pyro.optim.ClippedAdam({'lr':max_learning_rate,'lrd':(1-(5e-6))})
        elbo = pyro.infer.JitTraceEnum_ELBO(num_particles=num_particles,strict_enumeration_warning=False)
        svi = SVI(self.model, self.guide, scheduler, elbo)

        loss_tracker=[]
        self.train()
        self=self.to(device)
        self.set_approx(False)
        
        #for steps in range(max_steps):
        pbar = tqdm.tqdm(total=max_steps, position=0)
        done=False
        while steps < max_steps:
            for x in dataloader:
                x['step']=torch.ones(1).to(device)*steps
                x=[x[k].to(device) if k in x.keys() else torch.zeros(1) for k in self.args]
                loss=svi.step(*x)
                steps+=1
                if steps<max_steps-1:
                    #scheduler.step()
                    pass
                else:
                    break
                pbar.update(1)
                loss_tracker.append(loss)
                if steps%print_every == 0:
                    # Tell the scheduler we've done one epoch.
                    pbar.write("[Step %02d]  Loss: %.5f" % (steps, np.mean(loss_tracker[-print_every:])))
        
        pbar.close()
        allDone()
        print("Finished training!")