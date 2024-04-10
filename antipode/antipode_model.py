#Derived from testing version PBS1.9.1.8.1
import os
import sys
import sklearn
from sklearn import cluster
import pandas as pd
import scanpy as sc
import anndata
import scvi
import inspect
import tqdm
import numpy as np
import scipy
import gc
import torch
import torch.nn as nn
from torch.nn.functional import softplus, softmax
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.optim
import re
import inspect
from anndata import AnnData
from mudata import MuData
from typing import Literal, Optional
from scvi.module.base import PyroBaseModuleClass

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
        self.scale_factor=s2

    def prepare_phase_2(self,cluster='kmeans',prefix='',epochs = 5,device=None,dimension_reduction='X_antipode'):
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

        hierarchy=scipy.cluster.hierarchy.ward(kmeans_means)
        level_assignments=[scipy.cluster.hierarchy.cut_tree(hierarchy,n_clusters=x) for x in self.level_sizes]
        adj_means_dict=calculate_layered_tree_means(kmeans_means, level_assignments)
        new_clusts=[adj_means_dict[k][j] for k in adj_means_dict.keys() for j in adj_means_dict[k].keys()]
        new_locs=torch.tensor(new_clusts,device=device).float()
        
        edge_matrices=create_edge_matrices(level_assignments)
        edge_matrices=[torch.tensor(x,device=device) for x in edge_matrices]
        for i in range(len(self.level_sizes)-1):
            #pyro.get_param_store().__setitem__('edges_'+str(i), pyro.param('edges_'+str(i)).detach()+edge_matrices[i].T)
            pyro.get_param_store().__setitem__('edges_'+str(i), 1e-4*torch.randn(edge_matrices[i].T.shape,device=device).float()+edge_matrices[i].T.float())
        
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
        pyro.get_param_store().__setitem__('batch_dm',new_locs.new_zeros(pyro.param('batch_dm').shape))
        pyro.get_param_store().__setitem__('discov_di',new_locs.new_zeros(pyro.param('discov_di').shape))
        pyro.get_param_store().__setitem__('batch_di',new_locs.new_zeros(pyro.param('batch_di').shape))
        pyro.get_param_store().__setitem__('cluster_intercept',new_locs.new_zeros(pyro.param('cluster_intercept').shape))
    
    def common_training_loop(self, dataloader, max_steps, scheduler, svi, print_every, device, steps=0):
        self.losses = []
        pbar = tqdm.tqdm(total=max_steps, position=0)
        while steps < max_steps:
            for x in dataloader:
                x['step'] = torch.ones(1).to(device) * steps
                x = [x[k].squeeze(0).to(device) if k in x.keys() else torch.zeros(1) for k in self.args]
                if self.scale_factor == 1.:
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
                'lrd': (1 - (5e-6))
            })

    def train_phase(self, phase, max_steps, print_every=10000, device='cuda', max_learning_rate=0.001, num_particles=1, one_cycle_lr=True, steps=0, batch_size=32,freeze_encoder=None):
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
        elbo_class = pyro.infer.JitTrace_ELBO
        elbo = elbo_class(num_particles=num_particles, strict_enumeration_warning=False)
        hide_params=[name for name in pyro.get_param_store() if re.search('encoder',name)]
        guide=self.guide if not self.freeze_encoder else poutine.block(self.guide,hide=hide_params)
        svi = SafeSVI(self.model, guide, scheduler, elbo,clip_std_multiplier=6.0)  
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
                "the same order as the adata used to train the model.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
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


class ANTIPODE(PyroBaseModuleClass,AntipodeTrainingMixin, AntipodeSaveLoadMixin):
    """
    ANTIPODE (Single Cell Ancestral Node Taxonomy Inference by Parcellation of Differential Expression) 
    leverages variational inference for analyzing and categorizing cell types by accounting for biological and batch covariates and discrete and continuous latent variables. This model works by simultaneously integrating evolution-inspired differential expression parcellation, taxonomy generation (clustering) and batch correction.

    Parameters:
    adata (AnnData): The single-cell dataset encapsulated in an AnnData object.
    discov_pair (tuple): Specifies the discovery covariate's key and its location ('obs' or 'obsm') in the AnnData object.
    batch_pair (tuple): Specifies the batch covariate's key and its location ('obs' or 'obsm') in the AnnData object.
    layer (str): The specific layer of the AnnData object to be analyzed.
    level_sizes (list of int): Defines the hierarchical model structure (corresponding to a layered tree) by specifying the size of each level. Make sure each layer gets progressively larger and ideally start with a single root. Defaults to [1, 10, 100].
    bi_depth (int): Tree depth (from root) for batch identity effect correction. Defaults to 2. Should be less than length of level_sizes
    psi_levels (list of bool): Whether or not to allow a psi at each level of the layered tree. Should be length 1 (all levels) or len(level_sizes)
    
    num_latent (int): The number of latent dimensions to model. Defaults to 50.
    num_batch_embed (int): Number of embedding dimensions for batch effects. Defaults to 10. 
    scale_factor (float, optional): Factor for scaling the data normalization. Inferred from data if None. [DANGER]
    prior_scale (float): Scale for the Laplace prior distributions. Defaults to 100. [DANGER]
    dcd_prior (float, optional): Scale for discov_constitutive_de. Use this for missing genes (set to large negative value and rest 0. Zeros if None.
    use_psi (bool): Whether to utilize psi continuous variation parameter. Defaults to True.
    use_q_score (bool): Whether to use q continuous "quality" scores. Defaults to True.
    dist_normalize (bool): EXPERIMENTAL. Whether to apply distance normalization. Defaults to False.
    z_transform (pytorch function): Function to be applied to latent space (Z) e.g. centered_sigmoid, sigmoid. This will mess up DE Parameter scaling.
    loc_as_param, zdw_as_param, intercept_as_param (bool): Flags for using location, Z decoder weight, and intercept as parameters instead (maximum likelihood inference instead of Laplace MAP), respectively. All default to False.
    theta_prior (float): Initial value for the inverse dispersion of the negative binomial. Defaults to 50. [DANGER]
    scale_init_val (float): Initial value for scaling parameters. Defaults to 0.01. [DANGER]
    classifier_hidden, encoder_hidden, batch_embedder_hidden (list of int): Sizes of hidden layers for the classifier, encoder and batch embedding networks, respectively.
    sampler_category (string): Obs categorical column which will be used with the dataloader to sample each category with equal probability. (suggested use is the discov category)
    """

    def __init__(self, adata, discov_pair, batch_pair, layer, level_sizes=[1,10,100],
                 num_latent=50,scale_factor=None, prior_scale=100,dcd_prior=None,
                 use_psi=True,loc_as_param=False,zdw_as_param=False,intercept_as_param=False,use_q_score=True,psi_levels=[True],
                 num_batch_embed=10,theta_prior=50.,scale_init_val=0.01,bi_depth=2,dist_normalize=False,z_transform=None,
                 classifier_hidden=[3000,3000,3000],encoder_hidden=[6000,5000,3000,1000],batch_embedder_hidden=[1000,500,500],
                 sampler_category=None):

        pyro.clear_param_store()
        self.init_args = dict(locals())
        # Determine num_discov and num_batch from the AnnData object
        self.discov_loc, self.discov_key = discov_pair
        self.batch_loc, self.batch_key = batch_pair
        self.num_discov = adata.obsm[self.discov_key].shape[-1] if self.discov_loc == 'obsm' else len(adata.obs[self.discov_key].unique())
        self.num_batch = adata.obsm[self.batch_key].shape[-1] if self.batch_loc == 'obsm' else len(adata.obs[self.batch_key].unique())
        self.design_matrix = (self.discov_loc == 'obsm')
        self.layer = layer

        self._setup_adata_manager_store: dict[str, type[scvi.data.AnnDataManager]] = {}
        self.num_var = adata.layers[layer].shape[-1]
        self.num_latent = num_latent
        self.scale_factor = 1.0#scale_factor if scale_factor is not None else 2e2 / (self.num_var * num_particles * num_latent)
        self.num_batch_embed = num_batch_embed
        self.temperature = 0.1
        self.epsilon = 1e-5
        self.approx = False
        self.prior_scale = prior_scale
        self.use_psi = use_psi
        self.use_q_score = use_q_score
        self.loc_as_param = loc_as_param
        self.zdw_as_param = zdw_as_param
        self.intercept_as_param = intercept_as_param
        self.theta_prior = theta_prior
        self.scale_init_val = scale_init_val
        self.level_sizes = level_sizes
        self.num_labels = sum(level_sizes)
        self.bi_depth = bi_depth
        self.bi_depth = sum(self.level_sizes[:self.bi_depth])
        self.dist_normalize = dist_normalize
        self.sampler_category = sampler_category
        self.psi_levels = [float(x) for x in psi_levels]

        self.dcd_prior = torch.zeros((self.num_discov,self.num_var)) if dcd_prior is None else dcd_prior#Use this for 
                
        # Initialize plates to be used during sampling
        self.var_plate = pyro.plate('var_plate',self.num_var,dim=-1)
        self.discov_plate = pyro.plate('discov_plate',self.num_discov,dim=-3)
        self.batch_plate = pyro.plate('batch_plate',self.num_batch,dim=-3)
        self.latent_plate = pyro.plate('latent_plate',self.num_latent,dim=-1)
        self.latent_plate2 = pyro.plate('latent_plate2',self.num_latent,dim=-2)
        self.label_plate = pyro.plate('label_plate',self.num_labels,dim=-2)
        self.batch_embed_plate = pyro.plate('batch_embed_plate',self.num_batch_embed,dim=-3)
        self.bi_depth_plate = pyro.plate('bi_depth_plate',self.bi_depth,dim=-2)

        #Initialize MAP inference modules
        self.dm=MAPLaplaceModule(self,'discov_dm',[self.num_discov,self.num_labels,self.num_latent],[self.discov_plate,self.label_plate,self.latent_plate])
        self.bm=MAPLaplaceModule(self,'batch_dm',[self.num_batch,self.num_labels,self.num_latent],[self.batch_plate,self.label_plate,self.latent_plate])
        self.di=MAPLaplaceModule(self,'discov_di',[self.num_discov,self.num_labels,self.num_var],[self.discov_plate,self.label_plate,self.var_plate])
        self.bei=MAPLaplaceModule(self,'batch_di',[self.num_batch_embed,self.bi_depth,self.num_var],[self.batch_embed_plate,self.bi_depth_plate,self.var_plate])
        self.ci=MAPLaplaceModule(self,'cluster_intercept',[self.num_labels, self.num_var],[self.label_plate,self.var_plate],param_only=self.intercept_as_param)
        self.dc=MAPLaplaceModule(self,'discov_dc',[self.num_discov,self.num_latent,self.num_var],[self.discov_plate,self.latent_plate2,self.var_plate])
        self.zdw=MAPLaplaceModule(self,'z_decoder_weight',[self.num_latent,self.num_var],[self.latent_plate2,self.var_plate],init_val=((2/self.num_latent)*(torch.rand(self.num_latent,self.num_var)-0.5)),param_only=self.zdw_as_param)
        self.zl=MAPLaplaceModule(self,'locs',[self.num_labels,self.num_latent],[self.label_plate,self.latent_plate],param_only=self.loc_as_param)
        self.zs=MAPHalfCauchyModule(self,'scales',[self.num_labels,self.num_latent],[self.label_plate,self.latent_plate],init_val=self.scale_init_val*torch.ones(self.num_labels,self.num_latent),constraint=constraints.positive,param_only=False)
        self.zld=MAPLaplaceModule(self,'locs_dynam',[self.num_labels,self.num_latent],[self.label_plate,self.latent_plate],param_only=False)
        self.qg=MAPLaplaceModule(self,'quality_genes',[1,self.num_var],[self.var_plate],param_only=False)
        
        self.tree_edges = TreeEdges(self,straight_through=False)
        self.tree_convergence_bottom_up = TreeConvergenceBottomUp(self)        
        self.z_transform = null_function if z_transform is None else z_transform#centered_sigmoid#torch.special.expit

        if self.design_matrix:
            fields={'s':('layers',self.layer),
            'discov_ind':('obsm',self.discov_key),
            'batch_ind':('obsm',self.batch_key)}
            field_types={"s":np.float32,"batch_ind":np.float32,"discov_ind":np.float32}
        else:
            fields={'s':('layers',self.layer),
            'discov_ind':('obs',self.discov_key),
            'batch_ind':('obs',self.batch_key)}
            field_types={"s":np.float32,"batch_ind":np.int64,"discov_ind":np.int64}

        self.fields=fields
        self.field_types=field_types
        self.setup_anndata(adata, {'discov_ind': discov_pair, 'batch_ind': batch_pair}, self.field_types)
        
        super().__init__()
        # Setup the various neural networks used in the model and guide
        self.z_decoder=ZDecoder(num_latent=self.num_latent, num_var=self.num_var)        
        self.zl_encoder=ZLEncoder(num_var=self.num_var,hidden_dims=encoder_hidden,num_cat_input=self.num_discov,
                    outputs=[(self.num_latent,None),(self.num_latent,softplus),(1,None),(1,softplus)])
        
        self.classifier=Classifier(num_latent=self.num_latent,hidden_dims=classifier_hidden,
                    outputs=[(self.num_labels,None),(len(self.level_sizes),None),(len(self.level_sizes),softplus)])

        #Too large to exactly model gene-level batch effects for all cluster x batch
        self.be_nn=SimpleFFNN(in_dim=self.num_batch,hidden_dims=batch_embedder_hidden,
                    out_dim=self.num_batch_embed)
        
        self.epsilon = 0.006
        #Initialize model not in fuzzy mode
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

    def set_freeze_encoder(self,b: bool):
        self.freeze_encoder=b
        
    def model(self, s,discov_ind=torch.zeros(1),batch_ind=torch.zeros(1),step=torch.ones(1),taxon=torch.zeros(1),Z_obs=torch.zeros(1)):
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
        l = s.sum(1).unsqueeze(-1)
        
        # Scale all sample statements for numerical stability
        with poutine.scale(scale=self.scale_factor):
            # Counts parameter of NB (variance of the observation distribution)
            s_theta = pyro.param("s_inverse_dispersion", self.theta_prior * s.new_ones(self.num_var),
                               constraint=constraints.positive)
            #Weak overall histogram normalization
            discov_mul = pyro.param("discov_mul", s.new_ones(self.num_discov,1),constraint=constraints.positive) if self.dist_normalize else s.new_ones(self.num_discov,1)

            dcd=pyro.param("discov_constitutive_de", self.dcd_prior.to(s.device))
            level_edges=self.tree_edges.model_sample(s,approx=self.approx)
            quality_genes=self.qg.model_sample(s) if self.use_q_score else 0.
            
            with minibatch_plate:
                batch_embed=centered_sigmoid(pyro.sample('batch_embed', dist.Laplace(s.new_zeros(self.num_batch_embed),
                                self.prior_scale*s.new_ones(self.num_batch_embed),validate_args=True).to_event(1)))
                beta_prior_a=1.*s.new_ones(self.num_labels)
                beta_prior_a[0]=10. #0 block is consititutive
                if self.approx:#Bernoulli blocks approx?
                    taxon_probs = pyro.sample("taxon_probs", dist.Beta(beta_prior_a,s.new_ones(self.num_labels),validate_args=True).to_event(1))
                    taxon = pyro.sample('taxon',dist.RelaxedBernoulli(temperature=0.1*s.new_ones(1),probs=taxon_probs).to_event(1))
                else:
                    taxon_probs=pyro.sample('taxon_probs',dist.Dirichlet(s.new_ones(s.shape[0],self.level_sizes[-1]),validate_args=True))
                    if sum(taxon.shape) > 1:#Supervised?
                        if taxon.shape[-1]==self.num_labels:#Totally supervised?
                            pass
                        else:#Only bottom layer is supervised?
                            taxon = taxon_probs = pyro.sample("taxon", dist.OneHotCategorical(probs=taxon_probs,validate_args=True),obs=taxon)
                            taxon = self.tree_convergence_bottom_up.just_propagate(taxon,level_edges,s) if self.freeze_encoder else self.tree_convergence_bottom_up.just_propagate(taxon,level_edges,s)
                    else:#Unsupervised
                        taxon = pyro.sample("taxon", 
                                         model_distributions.SafeAndRelaxedOneHotCategorical(temperature=self.temperature*s.new_ones(1),probs=taxon_probs,validate_args=True))                    
                        taxon = self.tree_convergence_bottom_up.just_propagate(taxon,level_edges,s) if self.freeze_encoder else self.tree_convergence_bottom_up.just_propagate(taxon,level_edges,s)
                    taxon = torch.concat(taxon,-1)
                    taxon_probs=self.tree_convergence_bottom_up.just_propagate(taxon_probs[...,-self.level_sizes[-1]:],level_edges,s) if self.freeze_encoder else self.tree_convergence_bottom_up.just_propagate(taxon_probs[...,-self.level_sizes[-1]:],level_edges,s)
                    taxon_probs=torch.cat(taxon_probs,-1)
                   
            locs=self.zl.model_sample(s,scale=fest([taxon_probs],-1))
            scales=self.zs.model_sample(s,scale=fest([taxon_probs],-1))
            locs_dynam=self.zld.model_sample(s,scale=fest([taxon_probs],-1))
            discov_dm=self.dm.model_sample(s,scale=fest([discov,taxon_probs],-1))
            discov_di=self.di.model_sample(s,scale=fest([discov,taxon_probs],-1))
            batch_dm=self.bm.model_sample(s,scale=fest([batch,taxon_probs],-1))
            
            bei=self.bei.model_sample(s,scale=fest([batch_embed.abs(),taxon_probs[...,:self.bi_depth]],-1))
            cluster_intercept=self.ci.model_sample(s,scale=fest([taxon_probs],-1))
            
            with minibatch_plate:
                bi=torch.einsum('...bi,...ijk->...bjk',batch_embed,bei)
                bi=torch.einsum('...bj,...bjk->...bk',taxon[...,:self.bi_depth],bi)
                psi = centered_sigmoid(pyro.sample('psi',dist.Laplace(s.new_zeros(s.shape[0],len(self.level_sizes)),self.prior_scale*s.new_ones(s.shape[0],len(self.level_sizes))).to_event(1)))
                #psi = centered_sigmoid(pyro.sample('psi',dist.Logistic(s.new_zeros(s.shape[0],len(self.level_sizes)),s.new_ones(s.shape[0],len(self.level_sizes))).to_event(1)))
                psi=psi*torch.tensor(self.psi_levels).to(s.device).unsqueeze(0)
                psi = 0 if not self.use_psi or self.approx else torch.repeat_interleave(psi, torch.tensor(self.level_sizes).to(s.device), dim=1)
                q = torch.sigmoid(pyro.sample('q',dist.Logistic(s.new_zeros(s.shape[0],1),s.new_ones(s.shape[0],1)).to_event(1))) if self.use_q_score else 1.0
                this_locs=oh_index(locs,taxon)
                this_scales=oh_index(scales,taxon)
                z=pyro.sample('z_loc',dist.Laplace(this_locs,0.5*self.prior_scale*s.new_ones(s.shape[0],self.num_latent),validate_args=True).to_event(1))
                z_dist=dist.Normal(this_locs,this_scales+self.epsilon,validate_args=True).to_event(1)
                if sum(Z_obs.shape) <=1: 
                     z=pyro.sample('z', z_dist) 
                else: #Supervised latent space
                    z=pyro.sample('z', z_dist)
                    z=pyro.sample('z_obs', dist.Normal(z,this_scales+self.epsilon,validate_args=True).to_event(1),obs=Z_obs)

            cur_discov_dm = oh_index1(discov_dm, discov_ind) if self.design_matrix else discov_dm[discov_ind]
            cur_batch_dm = oh_index1(batch_dm, batch_ind) if self.design_matrix else batch_dm[batch_ind]
            cur_dcd = oh_index(dcd, discov) if self.design_matrix else  dcd[discov_ind]
            
            z=z+oh_index2(cur_discov_dm,taxon) + oh_index2(cur_batch_dm,taxon)+(oh_index(locs_dynam,taxon*psi))
            z=self.z_transform(z)                
            pseudo_z=oh_index(locs,taxon_probs)+oh_index2(discov_dm[discov_ind],taxon_probs) + oh_index2(batch_dm[batch_ind],taxon_probs)+(oh_index(locs_dynam,taxon_probs*psi))
            pseudo_z=self.z_transform(pseudo_z)
            z_decoder_weight=self.zdw.model_sample(s,scale=fest([pseudo_z.abs()],-1))
            discov_dc=self.dc.model_sample(s,scale=fest([discov,pseudo_z.abs()],-1))
            cur_discov_di = oh_index1(discov_di, discov_ind) if self.design_matrix else discov_di[discov_ind]
            cur_discov_dc = oh_index1(discov_dc, discov_ind) if self.design_matrix else discov_dc[discov_ind]
            cur_discov_di=oh_index2(cur_discov_di,taxon)
            cur_cluster_intercept=oh_index(cluster_intercept,taxon)
            
            mu=torch.einsum('...bi,...bij->...bj',z,z_decoder_weight+cur_discov_dc)#+bc
            spliced_mu=mu+cur_dcd+cur_discov_di+cur_cluster_intercept+bi+((1-q)*quality_genes)
            norm_spliced_mu=spliced_mu*discov_mul[discov_ind]
            spliced_out=softmax(norm_spliced_mu,dim=-1)
            log_mu = (l * spliced_out + 1e-6).log()
            
            with self.var_plate,minibatch_plate2:
                s_dist = dist.NegativeBinomial(total_count=(s_theta*q)+0.1,logits=log_mu-s_theta.log(),validate_args=True)
                s_out=pyro.sample("s", s_dist, obs=s.int())

    
    # The guide specifies the variational distribution
    def guide(self, s,discov_ind=torch.zeros(1),batch_ind=torch.zeros(1),step=torch.ones(1),taxon=torch.zeros(1),Z_obs=torch.zeros(1)):
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
        
        with poutine.scale(scale=self.scale_factor):
            level_edges=self.tree_edges.guide_sample(s,approx=self.approx) 
            with minibatch_plate:
                batch_embed=self.be_nn(batch)
                batch_embed=centered_sigmoid(pyro.sample('batch_embed', dist.Delta(batch_embed,validate_args=True).to_event(1)))
                if self.freeze_encoder:
                    with torch.no_grad():
                        z_loc, z_scale , q_loc,q_scale= self.zl_encoder(s,discov)
                        z_loc=z_loc.detach()
                        z_scale=z_scale.detach()
                        q_loc=q_loc.detach()
                        q_scale=q_scale.detach()
                else:
                    z_loc, z_scale, q_loc,q_scale= self.zl_encoder(s,discov)
                z=pyro.sample('z',dist.Normal(z_loc,z_scale+self.epsilon).to_event(1))
                q=pyro.sample('q',dist.Normal(q_loc,q_scale+self.epsilon).to_event(1))
                pyro.sample('z_loc',dist.Delta(z_loc).to_event(1))
                z=self.z_transform(z)
                taxon_logits,psi_loc,psi_scale=self.classifier(z)
                #psi=centered_sigmoid(pyro.sample('psi_loc',dist.Delta(psi_loc).to_event(1)))
                psi=centered_sigmoid(pyro.sample('psi',dist.Normal(psi_loc,psi_scale+self.epsilon).to_event(1)))
                psi=psi*torch.tensor(self.psi_levels).to(s.device).unsqueeze(0)
                psi = 0 if not self.use_psi or self.approx else torch.repeat_interleave(psi, torch.tensor(self.level_sizes).to(s.device), dim=1)
                if self.approx:
                    taxon_dist = dist.Delta(safe_sigmoid(taxon_logits),validate_args=True).to_event(1)
                    taxon_probs = pyro.sample("taxon_probs", taxon_dist)
                    taxon = pyro.sample('taxon',dist.RelaxedBernoulli(temperature=self.temperature*s.new_ones(1),probs=taxon_probs).to_event(1))
                else:
                    taxon_probs=pyro.sample('taxon_probs',dist.Delta(safe_softmax(taxon_logits[...,-self.level_sizes[-1]:],eps=1e-5)).to_event(1))
                    if sum(taxon.shape) > 1:
                        pass
                    else:
                        taxon = pyro.sample("taxon", 
                                         model_distributions.SafeAndRelaxedOneHotCategorical(temperature=self.temperature*s.new_ones(1),probs=taxon_probs,validate_args=True))                    
                    if taxon.shape[-1]<self.num_labels:
                        taxon = self.tree_convergence_bottom_up.just_propagate(taxon,level_edges,s) if self.freeze_encoder else self.tree_convergence_bottom_up.just_propagate(taxon,level_edges,s)
                        taxon = torch.concat(taxon,-1)
                    taxon_probs=self.tree_convergence_bottom_up.just_propagate(taxon_probs[...,-self.level_sizes[-1]:],level_edges,s) if self.freeze_encoder else self.tree_convergence_bottom_up.just_propagate(taxon_probs[...,-self.level_sizes[-1]:],level_edges,s)
                    taxon_probs=torch.cat(taxon_probs,-1)

            quality_genes=self.qg.guide_sample(s) if self.use_q_score else 0.
            locs=self.zl.guide_sample(s,scale=fest([taxon_probs],-1))
            scales=self.zs.guide_sample(s,scale=fest([taxon_probs],-1))
            locs_dynam=self.zld.guide_sample(s,scale=fest([taxon_probs],-1))
            discov_dm=self.dm.guide_sample(s,scale=fest([discov,taxon_probs],-1))
            batch_dm=self.bm.guide_sample(s,scale=fest([batch,taxon_probs],-1))
            discov_di=self.di.guide_sample(s,scale=fest([discov,taxon_probs],-1))
            cluster_intercept=self.ci.guide_sample(s,scale=fest([taxon_probs],-1))
            bei=self.bei.guide_sample(s,scale=fest([batch_embed.abs(),taxon_probs[...,:self.bi_depth]],-1))#maybe should be abs sum bei
            #For scaling
            if self.design_matrix:
                z=z+oh_index2(oh_index1(discov_dm,discov_ind),taxon) + oh_index2(oh_index1(batch_dm,batch_ind),taxon)+(oh_index(locs_dynam,taxon*psi))
            else:
                z=z+oh_index2(discov_dm[discov_ind],taxon) + oh_index2(batch_dm[batch_ind],taxon)+(oh_index(locs_dynam,taxon*psi))
            z=self.z_transform(z)
            pseudo_z=oh_index(locs,taxon_probs)+oh_index2(discov_dm[discov_ind],taxon_probs) + oh_index2(batch_dm[batch_ind],taxon_probs)+(oh_index(locs_dynam,taxon_probs*psi))
            pseudo_z=self.z_transform(pseudo_z)
            z_decoder_weight=self.zdw.guide_sample(s,scale=fest([pseudo_z.abs()],-1))
            discov_dc=self.dc.guide_sample(s,scale=fest([discov,pseudo_z.abs()],-1))
