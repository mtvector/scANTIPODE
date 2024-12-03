import os
import sys
import sklearn
from sklearn import cluster
import pandas as pd
import scanpy as sc
import anndata
import inspect
import tqdm
import numpy as np
import scipy
import gc
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.optim
import re
import inspect
from anndata import AnnData
from mudata import MuData
from typing import Literal, Optional
import scvi

from .model_modules import *
from .model_distributions import *
from .model_functions import *
from .train_utils import *
from .plotting import *


class AntipodeTrainingMixin:
    '''
    Mixin class providing functions to actually run ANTIPODE.
    The naive model trains in 3 phases, first a nonhierarchical block phase to estimate cell type manifolds, then phase 2 learns parameters for a fixed discrete clustering on a fixed latent space for initialization (or supervised). Phase 3 makes all parameters learnable.
    You can also use supervised taxonomy by providing a clustering as a discrete obsm matrix and training only phase2 with freeze_encoder=False.
    '''
    
    def save_params_to_uns(self,prefix=''):
        pstore=param_store_to_numpy()
        pstore={n:pstore[n] for n in pstore.keys() if not re.search('encoder|classifier|be_nn|\$\$\$',n)}
        pstore={n:pstore[n] for n in pstore.keys() if not np.isnan(pstore[n]).any()}
        self.adata_manager.adata.uns[prefix+'param_store']=pstore

    def get_antipode_outputs(self,batch_size=2048,device='cuda'):
        if 'discov_onehot' not in self.adata_manager.adata.obsm.keys():
            self.adata_manager.adata.obs[self.discov_key]=self.adata_manager.adata.obs[self.discov_key].astype('category')
            self.adata_manager.adata.obsm['discov_onehot']=numpy_onehot(self.adata_manager.adata.obs[self.discov_key].cat.codes)
        self.adata_manager.register_new_fields([scvi.data.fields.ObsmField('discov_onehot','discov_onehot')])
    
        field_types={"s":np.float32,"discov_onehot":np.float32}
        dataloader=scvi.dataloaders.AnnDataLoader(self.adata_manager,batch_size=32,drop_last=False,shuffle=False,data_and_attributes=field_types)#supervised_field_types for supervised step 
        encoder_outs=batch_output_from_dataloader(dataloader,self.zl_encoder,batch_size=batch_size,device=device)
        encoder_outs[0]=self.z_transform(encoder_outs[0])
        encoder_out=[x.detach().cpu().numpy() for x in encoder_outs]
        classifier_outs=batch_torch_outputs([(encoder_outs[0])],self.classifier,batch_size=batch_size,device='cuda')
        classifier_out=[x.detach().cpu().numpy() for x in classifier_outs]
        return encoder_out,classifier_out

    def store_outputs(self,device='cuda',prefix=''):
        self.save_params_to_uns(prefix='')
        self.to('cpu')
        self.eval()
        antipode_outs=self.get_antipode_outputs(batch_size=2048,device=device)
        self.allDone()
        taxon=antipode_outs[1][0]
        self.adata_manager.adata.obsm[prefix+'X_antipode']=antipode_outs[0][0]
        for i in range(antipode_outs[1][1].shape[1]):
            self.adata_manager.adata.obs[prefix+'psi_'+str(i)]=numpy_centered_sigmoid(antipode_outs[1][1][...,i])
        self.adata_manager.adata.obs[prefix+'q_score']=scipy.special.expit(antipode_outs[0][2])
        level_edges=[numpy_hardmax(self.adata_manager.adata.uns[prefix+'param_store']['edges_'+str(i)],axis=-1) for i in range(len(self.level_sizes)-1)]
        levels=self.tree_convergence_bottom_up.just_propagate(scipy.special.softmax(taxon[...,-self.level_sizes[-1]:],axis=-1),level_edges,s=torch.ones(1))
        prop_taxon=np.concatenate(levels,axis=-1)
        self.adata_manager.adata.obsm[prefix+'taxon_probs']=prop_taxon
        levels=self.tree_convergence_bottom_up.just_propagate(numpy_hardmax(levels[-1],axis=-1),level_edges,s=torch.ones(1))
        for i in range(len(levels)):
            cur_clust=prefix+'level_'+str(i)
            self.adata_manager.adata.obs[cur_clust]=levels[i].argmax(1)
            self.adata_manager.adata.obs[cur_clust]=self.adata_manager.adata.obs[cur_clust].astype(str)
        self.adata_manager.adata.obs[prefix+'antipode_cluster'] = self.adata_manager.adata.obs.apply(lambda x: '_'.join([x[prefix+'level_'+str(i)] for i in range(len(levels))]), axis=1)
        self.adata_manager.adata.obs[prefix+'antipode_cluster'] = self.adata_manager.adata.obs[prefix+'antipode_cluster'].astype(str)    
    
    def pretrain_classifier(self,epochs = 5,learning_rate = 0.001,batch_size = 64,prefix='',cluster='kmeans',device='cuda'):
        '''basic pytorch training of feed forward classifier to ease step 2'''        
        self.train()
        
        model = self.classifier.to(device)
        input_tensor =  torch.tensor(self.adata_manager.adata.obsm[self.dimension_reduction])  # Your input features tensor, shape [n_samples, n_features]
        target_tensor = torch.tensor(self.adata_manager.adata.obsm[cluster+'_onehot'])  # Your target labels tensor, shape [n_samples]    
        
        # Step 1: Prepare to train
        dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        #Training loop
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                # Forward pass
                outputs = model(inputs.to(device))
                loss = criterion(softmax(outputs[0],-1)[:,-targets.shape[-1]:], targets.to(device))
        
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')     

    def fix_scale_factor(self,svi,x,ideal_val=0.1):
        o1=svi.evaluate_loss(*x)
        s1=self.scale_factor
        s2=ideal_val*s1/o1
        self.scale_factor = np.absolute(s2)

    def prepare_phase_2(self,cluster='kmeans',prefix='',epochs = 5,device=None,dimension_reduction='X_antipode',reset_dc=True,naive_init=False):
        '''Run this if not running in supervised only mode (JUST phase2 with provided obsm clustering), 
        runs kmeans if cluster=kmeans, else uses the obs column provided by cluster. epochs=None skips pretraing of classifier
        To learn a latent space from scratch set dimension_reduction to None and use freeze_encoder=False'''
        if cluster=='kmeans':
            kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=self.level_sizes[-1],init='k-means++',max_iter=1000,reassignment_ratio=0.001,n_init=100,random_state=0).fit(self.adata_manager.adata.obsm[dimension_reduction])
            self.adata_manager.adata.obs['kmeans']=kmeans.labels_
            self.adata_manager.adata.obs['kmeans']=self.adata_manager.adata.obs['kmeans'].astype(int).astype('category')
            self.adata_manager.adata.obsm['kmeans_onehot']=numpy_onehot(self.adata_manager.adata.obs['kmeans'].cat.codes,num_classes=self.level_sizes[-1])
        else:
            self.adata_manager.adata.obs[cluster]=self.adata_manager.adata.obs[cluster].astype('category')
            self.adata_manager.adata.obsm[cluster+'_onehot']=numpy_onehot(self.adata_manager.adata.obs[cluster].cat.codes,num_classes=self.level_sizes[-1])
        device=pyro.param('locs').device if device is None else device
        self.adata_manager.register_new_fields([make_field('taxon',('obsm',cluster+'_onehot'))])
        if dimension_reduction is not None:#For supervised Z register dr
            self.dimension_reduction=dimension_reduction
            self.adata_manager.register_new_fields([make_field('Z_obs',('obsm',dimension_reduction))])
        if (epochs is not None) and (dimension_reduction is not None):
            self.pretrain_classifier(cluster=cluster,prefix=prefix,epochs=epochs,device=device)
        kmeans_means=group_aggr_anndata(self.adata_manager.adata,[cluster], agg_func=np.mean,layer=dimension_reduction,obsm=True)[0]
        if 'locs' not in [x for x in pyro.get_param_store()]:
            print('quick init')
            self.train_phase(phase=1,max_steps=1,print_every=10000,num_particles=1,device=device, max_learning_rate=1e-10, one_cycle_lr=True, steps=0, batch_size=4)
            self.cpu()
            
        if naive_init:
            new_locs=torch.concatenate(
                [pyro.param('locs').new_zeros(sum(self.level_sizes[:-1]),pyro.param('locs').shape[1]),
                 torch.tensor(kmeans_means-kmeans_means.mean(0),device=pyro.param('locs').device).float()],
                 axis=0).float()
            new_locs[0,:]=torch.tensor(kmeans_means.mean(0)).float()
        else:
            hierarchy=scipy.cluster.hierarchy.ward(kmeans_means)
            level_assignments=[scipy.cluster.hierarchy.cut_tree(hierarchy,n_clusters=x) for x in self.level_sizes]
            adj_means_dict=calculate_layered_tree_means(kmeans_means, level_assignments)
            new_clusts=[adj_means_dict[k][j] for k in adj_means_dict.keys() for j in adj_means_dict[k].keys()]
            new_locs=torch.tensor(new_clusts,device=device).float()
        
        edge_matrices=create_edge_matrices(level_assignments)
        edge_matrices=[torch.tensor(x,device=device) for x in edge_matrices]
        for i in range(len(self.level_sizes)-1):
            #pyro.get_param_store().__setitem__('edges_'+str(i), pyro.param('edges_'+str(i)).detach()+edge_matrices[i].T)
            pyro.get_param_store().__setitem__('edges_'+str(i), 1e-4 * torch.randn(edge_matrices[i].T.shape,device=device).float() + edge_matrices[i].T.float())
        
        self.adata_manager.adata.obs[cluster].astype(int)
        new_scales=group_aggr_anndata(self.adata_manager.adata,[cluster], agg_func=np.std,layer=dimension_reduction,obsm=True)[0]
        new_scales=torch.concatenate(
            [1e-5 * self.scale_init_val * new_locs.new_ones(sum(self.level_sizes[:-1]), pyro.param('locs').shape[1],requires_grad=True),
             torch.tensor(new_scales+1e-10,device=device,requires_grad=True)],axis=0).float()
        self.adata_manager.adata.obs[cluster].astype(str)
        pyro.get_param_store().__setitem__('locs',new_locs)
        pyro.get_param_store().__setitem__('locs_dynam',new_locs.new_zeros(new_locs.shape))
        pyro.get_param_store().__setitem__('scales',new_scales)
        self.adata_manager.adata.obs[cluster]=self.adata_manager.adata.obs[cluster].astype(str)
        pyro.get_param_store().__setitem__('discov_dm',new_locs.new_zeros(pyro.param('discov_dm').shape))
        pyro.get_param_store().__setitem__('seccov_dm',new_locs.new_zeros(pyro.param('seccov_dm').shape))
        pyro.get_param_store().__setitem__('batch_dm',new_locs.new_zeros(pyro.param('batch_dm').shape))
        pyro.get_param_store().__setitem__('discov_di',new_locs.new_zeros(pyro.param('discov_di').shape))
        pyro.get_param_store().__setitem__('batch_di',new_locs.new_zeros(pyro.param('batch_di').shape))
        pyro.get_param_store().__setitem__('cluster_intercept',new_locs.new_zeros(pyro.param('cluster_intercept').shape))
        if reset_dc: #DC doesn't necessarily need to be reset, can explode challenging models
            pyro.get_param_store().__setitem__('discov_dc',new_locs.new_zeros(pyro.param('discov_dc').shape))
    
    def common_training_loop(self, dataloader, max_steps, scheduler, svi, print_every, device, steps=0):
        self.losses = []
        pbar = tqdm.tqdm(total=max_steps, position=0)
        while steps < max_steps:
            for x in dataloader:
                x['step'] = torch.ones(1).to(device) * steps
                x = [x[k].squeeze(0).to(device) if k in x.keys() else torch.zeros(1) for k in self.args]
                if (self.scale_factor == 1.) or (steps == 2000):
                    print('fix scale factor')
                    self.fix_scale_factor(svi, x)
                pbar.update(1)
                loss = svi.step(*x)
                steps += 1
                if hasattr(scheduler, 'step'):
                    scheduler.step()
                if steps >= max_steps - 1 :
                    break
                
                self.losses.append(loss)
                if steps % print_every == 0:
                    pbar.write(f"[Step {steps:02d}]  Loss: {np.mean(self.losses[-print_every:]):.5f}")
        pbar.close()
        try:
            self.allDone()
        except:
            pass

    def setup_scheduler(self, max_learning_rate, max_steps, one_cycle_lr):
        if one_cycle_lr:
            return pyro.optim.OneCycleLR({
                'max_lr': max_learning_rate,
                'total_steps': max_steps,
                'div_factor': 100,
                'optim_args': {},
                'optimizer': torch.optim.Adam
            })
        else:
            return pyro.optim.ClippedAdam({
                'lr': max_learning_rate,
                'lrd': (1 - (1e-6))
            })

    def train_phase(self, phase, max_steps, print_every=10000, device='cuda', max_learning_rate=0.001, num_particles=1, one_cycle_lr=True, steps=0, batch_size=32,freeze_encoder=None,print_elbo=False,clip_std=6.0):
        self.scale_factor=1.
        freeze_encoder = True if freeze_encoder is None and phase == 2 else freeze_encoder
        freeze_encoder = False if freeze_encoder is None else  freeze_encoder
        self.set_freeze_encoder(freeze_encoder) 
        supervised_field_types=self.field_types.copy()
        supervised_fields=self.fields.copy()
        supervised_field_types["taxon"]=np.float32
        
        if not freeze_encoder and ("Z_obs" in [x.registry_key for x in  self.adata_manager.fields]) and phase == 2: #Running supervised D.R. (can't freeze encoder and run d.r.)
            supervised_field_types["Z_obs"]=np.float32
        field_types=self.field_types if phase != 2 else supervised_field_types
        sampler=create_weighted_random_sampler(self.adata_manager.adata.obs[self.sampler_category]) if self.sampler_category is not None else create_weighted_random_sampler(pd.Series(["same_category"] * self.adata_manager.adata.shape[0]))
        sampler= torch.utils.data.BatchSampler(sampler=sampler,batch_size=batch_size,drop_last=True)
        dataloader = scvi.dataloaders.AnnDataLoader(self.adata_manager, batch_size=batch_size, drop_last=True, sampler=sampler, data_and_attributes=field_types)
        scheduler = self.setup_scheduler(max_learning_rate, max_steps, one_cycle_lr)
        elbo_class = pyro.infer.JitTrace_ELBO if not print_elbo else Print_Trace_ELBO
        elbo = elbo_class(num_particles=num_particles, strict_enumeration_warning=False)
        hide_params=[name for name in pyro.get_param_store() if re.search('encoder',name)]
        guide=self.guide if not self.freeze_encoder else poutine.block(self.guide,hide=hide_params)
        svi = SafeSVI(self.model, guide, scheduler, elbo,clip_std_multiplier=clip_std)  
        self.train()
        self.zl_encoder.eval() if self.freeze_encoder else self.zl_encoder.train()
        self = self.to(device)
        self.set_approx(phase == 1)
        return self.common_training_loop(dataloader, max_steps, scheduler, svi, print_every, device, steps)
        
    def allDone(self):
        print("Finished training!")
        self.to('cpu')
        try:
            import IPython
            from IPython.display import Audio, display
            IPython.display.clear_output()#Make compatible with jupyter nbconvert
            display(Audio(url='https://notification-sounds.com/soundsfiles/Meditation-bell-sound.mp3', autoplay=True))
        except:
            pass
    
    def clear_cuda(self):
        '''Throw the kitchen sink at clearing the cuda cache for jupyter notebooks. 
        Might want to wrap in tryexcept'''
        import traceback
        self.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()
        try:
            a = 1/0 
        except Exception as e:  
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.clear_frames(exc_traceback)

class AntipodeSaveLoadMixin:
    '''Directly taken and modified from scvi-tools base_model and auxiliary functions'''
    def _get_user_attributes(self):
        """Returns all the self attributes defined in a model class, e.g., `self.is_trained_`."""
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))]
        attributes = [a for a in attributes if not a[0].startswith("_abc_")]
        return attributes

    @classmethod
    def _initialize_model(cls, adata, attr_dict,param_store_path,device):
        """Helper to initialize a model."""
        try:
            attr_dict.pop('__class__')
        except:
            pass
        model = cls(adata, **attr_dict)
        
        pyro.get_param_store().load(param_store_path,map_location=device)
        for k in list(pyro.get_param_store()):
            if '$$$' in k:
                pyro.get_param_store().__delitem__(k)
        return model

    def save(
        self,
        dir_path: str,
        prefix: str | None = None,
        overwrite: bool = False,
        save_anndata: bool = False,
        save_kwargs: dict | None = None,
        **anndata_write_kwargs,
    ):
        """Save the state of the model.

        Neither the trainer optimizer state nor the trainer history are saved.
        Model files are not expected to be reproducibly saved and loaded across versions
        until we reach version 1.0.

        Parameters
        ----------
        dir_path
            Path to a directory.
        prefix
            Prefix to prepend to saved file names.
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_anndata
            If True, also saves the anndata
        save_kwargs
            Keyword arguments passed into :func:`~torch.save`.
        anndata_write_kwargs
            Kwargs for :meth:`~anndata.AnnData.write`
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=overwrite)

        file_name_prefix = prefix or ""
        save_kwargs = save_kwargs or {}

        model_save_path = os.path.join(dir_path, f"{file_name_prefix}model.pt")

        # save the model state dict and the trainer state dict only
        model_state_dict = self.state_dict()
        
        var_names = self.adata_manager.adata.var_names.astype(str)
        var_names = var_names.to_numpy()

        user_attributes = self.init_args
        try:
            user_attributes.pop('adata')
            user_attributes.pop('self')
        except:
            pass
            
        pyro.get_param_store().save(os.path.join(dir_path,prefix+'antipode.paramstore'))

        torch.save(
            {
                "model_state_dict": model_state_dict,
                "var_names": var_names,
                "attr_dict": user_attributes,
            },
            model_save_path,
            **save_kwargs,
        )
        
        if save_anndata:
            file_suffix = ""
            if isinstance(self.adata_manager.adata, AnnData):
                file_suffix = "adata.h5ad"
            elif isinstance(self.adata_manager.adata, MuData):
                file_suffix = "mdata.h5mu"
            self.adata_manager.adata.write_h5ad(
                os.path.join(dir_path, f"{file_name_prefix}{file_suffix}"),
                **anndata_write_kwargs,
            )


    @classmethod
    def _validate_var_names(cls,adata, source_var_names):
        user_var_names = adata.var_names.astype(str)
        if not np.array_equal(source_var_names, user_var_names):
            warnings.warn(
                "var_names for adata passed in does not match var_names of adata used to "
                "train the model. For valid results, the vars need to be the same and in "
                "the same order as the adata used to train the model."#,
                #UserWarning,
                #stacklevel=settings.warnings_stacklevel,
            )    
    
    @classmethod
    def _load_saved_files(
        cls,
        dir_path: str,
        load_adata: bool,
        prefix: Optional[str] = None,
        is_mudata = False,
        load_kw_args = {'backed':'r'}
    ) -> tuple[dict, np.ndarray, dict, AnnData]:
        """Helper to load saved files."""
        file_name_prefix = prefix or ""
    
        model_file_name = f"{file_name_prefix}model.pt"
        model_path = os.path.join(dir_path, model_file_name)
        try:
            model = torch.load(model_path)
        except FileNotFoundError as exc:
            raise ValueError(
                f"Failed to load model file at {model_path}. "
                "If attempting to load a saved model from <v0.15.0, please use the util function "
                "`convert_legacy_save` to convert to an updated format."
            ) from exc
    
        model_state_dict = model["model_state_dict"]
        var_names = model["var_names"]
        attr_dict = model["attr_dict"]
    
        if load_adata:
            file_suffix = "adata.h5ad"
            adata_path = os.path.join(dir_path, f"{file_name_prefix}{file_suffix}")
            if os.path.exists(adata_path):
                if is_mudata:
                    adata = mudata.read(adata_path,**load_kw_args)
                else:
                    adata = anndata.read_h5ad(adata_path,**load_kw_args)
            else:
                raise ValueError("Save path contains no saved anndata and no adata was passed.")
        else:
            adata = None

        return attr_dict, var_names, model_state_dict, adata
        
    @classmethod
    def load(
        cls,
        dir_path: str,
        adata = None,
        accelerator: str = "auto",
        device: int | str = "auto",
        prefix: str | None = None,
        is_mudata: bool = False, 
        load_kw_args = {'backed':'r'}
    ):
        """Instantiate a model from the saved output.

        Parameters
        ----------
        dir_path
            Path to saved outputs.
        adata
            AnnData organized in the same way as data used to train model.
            It is not necessary to run setup_anndata,
            as AnnData is validated against the saved `scvi` setup dictionary.
            If None, will check for and load anndata saved with the model.
        %(param_accelerator)s
        %(param_device)s
        prefix
            Prefix of saved file names.

        Returns
        -------
        Model with loaded state dictionaries.

        Examples
        --------
        >>> model = ModelClass.load(save_path, adata)
        >>> model.get_....
        """
        load_adata = adata is None

        (
            attr_dict,
            var_names,
            model_state_dict,
            new_adata,
        ) = cls._load_saved_files(
            dir_path,
            load_adata,
            prefix=prefix,
            is_mudata=is_mudata,
            load_kw_args=load_kw_args,
        )
        
        adata = new_adata if new_adata is not None else adata

        cls._validate_var_names(adata, var_names)
        
        model = cls._initialize_model(adata, attr_dict,os.path.join(dir_path,prefix+'antipode.paramstore'),device=device)
        model.load_state_dict(model_state_dict)
        model.eval()
        #,os.path.join(dir_path,'antipode.paramstore')
        #model._validate_anndata(adata)
        return model


#########DEBUGGING########


import pyro
import pyro.ops.jit
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import (
    MultiFrameTensor,
    get_plate_stacks,
    is_validation_enabled,
    torch_item,
)
from pyro.util import check_if_enumerated, warn_if_nan


def _compute_log_r(model_trace, guide_trace):
    log_r = MultiFrameTensor()
    stacks = get_plate_stacks(model_trace)
    for name, model_site in model_trace.nodes.items():
        if model_site["type"] == "sample":
            log_r_term = model_site["log_prob"]
            if not model_site["is_observed"]:
                log_r_term = log_r_term - guide_trace.nodes[name]["log_prob"]
            log_r.add((stacks[name], log_r_term.detach()))
    return log_r



class Print_Trace_ELBO(ELBO):
    """
    A trace implementation of ELBO-based SVI. The estimator is constructed
    along the lines of references [1] and [2]. There are no restrictions on the
    dependency structure of the model or the guide. The gradient estimator includes
    partial Rao-Blackwellization for reducing the variance of the estimator when
    non-reparameterizable random variables are present. The Rao-Blackwellization is
    partial in that it only uses conditional independence information that is marked
    by :class:`~pyro.plate` contexts. For more fine-grained Rao-Blackwellization,
    see :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO`.

    References

    [1] Automated Variational Inference in Probabilistic Programming,
        David Wingate, Theo Weber

    [2] Black Box Variational Inference,
        Rajesh Ranganath, Sean Gerrish, David M. Blei
    """

    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, args, kwargs
        )
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            elbo_particle = torch_item(model_trace.log_prob_sum()) - torch_item(
                guide_trace.log_prob_sum()
            )
            elbo += elbo_particle / self.num_particles

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss

    def _differentiable_loss_particle(self, model_trace, guide_trace):
        elbo_particle = 0
        surrogate_elbo_particle = 0
        log_r = None

        # compute elbo and surrogate elbo
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                print(name,site["log_prob_sum"])
                elbo_particle = elbo_particle + torch_item(site["log_prob_sum"])
                surrogate_elbo_particle = surrogate_elbo_particle + site["log_prob_sum"]

        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                log_prob, score_function_term, entropy_term = site["score_parts"]
                print(name,site["log_prob_sum"])
                elbo_particle = elbo_particle - torch_item(site["log_prob_sum"])

                if not is_identically_zero(entropy_term):
                    surrogate_elbo_particle = (
                        surrogate_elbo_particle - entropy_term.sum()
                    )

                if not is_identically_zero(score_function_term):
                    if log_r is None:
                        log_r = _compute_log_r(model_trace, guide_trace)
                    site = log_r.sum_to(site["cond_indep_stack"])
                    surrogate_elbo_particle = (
                        surrogate_elbo_particle + (site * score_function_term).sum()
                    )

        return -elbo_particle, -surrogate_elbo_particle

    def differentiable_loss(self, model, guide, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the model and guide parameters
        """
        loss = 0.0
        surrogate_loss = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(
                model_trace, guide_trace
            )
            surrogate_loss += surrogate_loss_particle / self.num_particles
            loss += loss_particle / self.num_particles
        warn_if_nan(surrogate_loss, "loss")
        return loss + (surrogate_loss - torch_item(surrogate_loss))

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        loss = 0.0
        # grab a trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(
                model_trace, guide_trace
            )
            loss += loss_particle / self.num_particles

            # collect parameters to train from model and guide
            trainable_params = any(
                site["type"] == "param"
                for trace in (model_trace, guide_trace)
                for site in trace.nodes.values()
            )

            if trainable_params and getattr(
                surrogate_loss_particle, "requires_grad", False
            ):
                surrogate_loss_particle = surrogate_loss_particle / self.num_particles
                surrogate_loss_particle.backward(retain_graph=self.retain_graph)
        warn_if_nan(loss, "loss")
        return loss
