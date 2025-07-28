import os
import sys
import sklearn
from sklearn import cluster
import pandas as pd
import scanpy as sc
import anndata
from anndata import AnnData
from mudata import MuData
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
    The naive model trains in 3 phases, first a nonhierarchical fuzzy phase to estimate cell type manifolds. 
    Then phase 2 learns parameters for a fixed discrete clustering on a fixed latent space for initialization (or supervised). 
    Phase 3 makes all parameters learnable.
    You can also use supervised taxonomy by providing a clustering as a discrete obsm matrix and training only phase2 with freeze_encoder=False.
    '''
    
    def save_params_to_uns(self,prefix=''):
        pstore=param_store_to_numpy()
        pstore={n:pstore[n] for n in pstore.keys() if not re.search('encoder|classifier|be_nn|\\$\\$\\$',n)}
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
        # self.allDone()
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
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        #Training loop
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                if inputs.size(0) < 2:
                    print("Skipping batch of size 1 to avoid BatchNorm issues.")
                    continue
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
            # scipy.cluster.hierarchy.linkage(kmeans_means,method='average',metric='cityblock')
            level_assignments=[scipy.cluster.hierarchy.cut_tree(hierarchy,n_clusters=x) for x in self.level_sizes]
            adj_means_dict=calculate_layered_tree_means(kmeans_means, level_assignments)
            new_clusts=[adj_means_dict[k][j] for k in adj_means_dict.keys() for j in adj_means_dict[k].keys()]
            new_locs=torch.tensor(new_clusts,device=device).float()
        
        edge_matrices=create_edge_matrices(level_assignments)
        edge_matrices=[torch.tensor(x,device=device) for x in edge_matrices]
        for i in range(len(self.level_sizes)-1):
            #pyro.get_param_store().__setitem__('edges_'+str(i), pyro.param('edges_'+str(i)).detach()+edge_matrices[i].T)
            pyro.get_param_store().__setitem__('edges_'+str(i), 1e-4 * torch.randn(edge_matrices[i].T.shape,device=device).float() + edge_matrices[i].T.float())
        
        self.adata_manager.adata.obs[cluster].astype(str)
        new_scales=group_aggr_anndata(self.adata_manager.adata,[cluster], agg_func=np.std,layer=dimension_reduction,obsm=True)[0]
        new_scales=torch.concatenate(
            [1e-5 * self.scale_init_val * new_locs.new_ones(sum(self.level_sizes[:-1]), pyro.param('locs').shape[1],requires_grad=True),
             torch.tensor(new_scales+1e-10,device=device,requires_grad=True)],axis=0).float()
        
        pyro.get_param_store().__setitem__('locs',new_locs)
        pyro.get_param_store().__setitem__('locs_dynam',1e-5*torch.randn(new_locs.shape,device=new_locs.device))
        pyro.get_param_store().__setitem__('scales',new_scales)
        self.adata_manager.adata.obs[cluster]=self.adata_manager.adata.obs[cluster].astype(str)
        pyro.get_param_store().__setitem__('discov_dm',1e-5*torch.randn(pyro.param('discov_dm').shape,device=new_locs.device))
        pyro.get_param_store().__setitem__('seccov_dm',1e-5*torch.randn(pyro.param('seccov_dm').shape,device=new_locs.device))
        pyro.get_param_store().__setitem__('batch_dm',1e-5*torch.randn(pyro.param('batch_dm').shape,device=new_locs.device))
        pyro.get_param_store().__setitem__('discov_di',1e-5*torch.randn(pyro.param('discov_di').shape,device=new_locs.device))
        pyro.get_param_store().__setitem__('batch_di',1e-5*torch.randn(pyro.param('batch_di').shape,device=new_locs.device))
        pyro.get_param_store().__setitem__('cluster_intercept',1e-5*torch.randn(pyro.param('cluster_intercept').shape,device=new_locs.device))
        if reset_dc: #DC doesn't necessarily need to be reset, can explode challenging models
            pyro.get_param_store().__setitem__('discov_dc',1e-5*torch.randn(pyro.param('discov_dc').shape,device=new_locs.device))
    
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
        freeze_encoder = False if freeze_encoder is None else freeze_encoder
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

    @classmethod
    def load_and_recorrect_standard(cls,model_path,batch_size=128,n_steps=None,device='cuda'):
        '''convenience function for when I change correction code after finishing the model ':)'''
        adata = sc.read_h5ad(os.path.join(model_path,'p4_adata.h5ad'),backed='r')
        antipode_model = cls.load(model_path,adata=adata,prefix='p3_',device=device)
        antipode_model.store_outputs(device=device,prefix='')
        antipode_model.to(device)
        if n_steps is not None:
            posterior_out,posterior_categories = antipode_model.correct_fits(batch_size=batch_size, n_steps = n_steps)
        else:
            print('running intercept correction')
            posterior_out, posterior_categories = antipode_model.correct_fits_intercepts(batch_size=batch_size)
        antipode_model.store_outputs(device=device,prefix='')
        return antipode_model

    
    def run_standard_protocol(self, out_path, max_steps=500000, num_particles=3,
                              device='cuda', max_learning_rate=1e-3, one_cycle_lr=True, 
                              batch_size=32, correction_steps=None):
        if isinstance(max_steps, int):
            max_steps = [max_steps] * 3  # Now each phase uses the same value.
        elif isinstance(max_steps, list) and len(max_steps) == 3:
            pass  # Already a list of 3 values, so use as-is.
        # Map phases to checkpoint file paths
        checkpoint_files = {
            1: os.path.join(out_path, 'p1_model.pt'),
            2: os.path.join(out_path, 'p2_model.pt'),
            3: os.path.join(out_path, 'p3_model.pt'),
        }
        # Determine the highest completed phase (default to 0 if none found)
        last_completed_phase = max(
            (phase for phase, path in checkpoint_files.items() if os.path.exists(path)), 
            default=0
        )
        leaf_level = 'level_' + str(len(self.level_sizes)-1)
        print('last completed phase:', last_completed_phase)
        # Check if p4_adata.h5ad exists; if so, load it and use it for loading the model.
        p4adata_path = os.path.join(out_path, 'p4_adata.h5ad')
        if os.path.exists(p4adata_path):
            # Load the adata exactly as in your manual workflow.
            try:
                del self.adata_manager.adata
                del adata
            except:
                pass
            print("automatically using ",p4adata_path) #weird unsolveable bug when just skipping to 4 below
            self = self.load_and_recorrect_standard(out_path,batch_size=128, n_steps = correction_steps,device=device) #use this func instead
            try:
                self.save(out_path, save_anndata=True, prefix='p4_recorrect_')
            except Exception as e:
                print("saving error ocurred:",e)
            return self
        else:
            # Otherwise, use the adata already in the model.
            adata = self.adata_manager.adata
        # If a checkpoint exists, load the corresponding model with the chosen adata.
        if last_completed_phase:
            self.load(out_path, prefix=f'p{last_completed_phase}_', adata=adata, device=device)
            print(f"Resuming from phase {last_completed_phase}")
        else:
            print("No checkpoints found. Starting training from scratch.")

        # Execute remaining phases
        if last_completed_phase < 1:
            # Phase 1
            print('Running phase 1')
            self.train_phase(phase=1, max_steps=max_steps[0], print_every=10000, num_particles=num_particles,
                             device=device, max_learning_rate=max_learning_rate, one_cycle_lr=one_cycle_lr,
                             batch_size=batch_size, clip_std=100.)
            plot_loss(self.losses)
            self.store_outputs(device=device, prefix='')
            self.clear_cuda()
            self.save(out_path, save_anndata=False, prefix='p1_')
        if last_completed_phase < 2:
            # Phase 2
            print('Running phase 2')
            self.store_outputs(device=device, prefix='')
            self.prepare_phase_2(epochs=2, device=device, dimension_reduction='X_antipode')
            self.train_phase(phase=2, max_steps=max_steps[1], print_every=10000, num_particles=num_particles,
                             device=device, max_learning_rate=max_learning_rate, one_cycle_lr=one_cycle_lr,
                             batch_size=batch_size, freeze_encoder=True, clip_std=100.)
            plot_loss(self.losses)
            self.store_outputs(device=device, prefix='')
            self.clear_cuda()
            self.save(out_path, save_anndata=False, prefix='p2_')
        
        if last_completed_phase < 3:
            # Phase 3
            print('Running phase 3')
            self.store_outputs(device=device, prefix='')
            self.train_phase(phase=3, max_steps=max_steps[2], print_every=10000, num_particles=num_particles,
                             device=device, max_learning_rate=max_learning_rate, one_cycle_lr=one_cycle_lr,
                             batch_size=batch_size, clip_std=100)
            plot_loss(self.losses)
            self.store_outputs(device=device, prefix='')
            self.clear_cuda()
            self.save(out_path, save_anndata=True, prefix='p3_')
            self.to(device)
        self.to(device)
        self.eval()
        self.store_outputs(device=device, prefix='')#Cheap and just for safety
        # Final correction (always run after phase 3)
        print('Running final correction')
        try:
            if correction_steps is not None:
                posterior_out, posterior_categories = self.correct_fits(batch_size=128, n_steps=correction_steps,num_particles=num_particles)
            else:
                posterior_out, posterior_categories = self.correct_fits_intercepts(batch_size=128)
                
            self.store_outputs(device=device, prefix='')
            try:
                self.save(out_path, save_anndata=False, prefix='p4_')
            except Exception as e:
                print("saving error ocurred:",e)
        except Exception as e:
            print("Skipping correction",e)
        return self
    
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

    def calculate_cluster_params(self, flavor:['torch','numpy'] = 'numpy',prefix='',cluster_count_threshold=50):
        adata = self.adata_manager.adata
        leaf_level = 'level_' + str(len(self.level_sizes)-1)
        if flavor == 'torch':
            pstore=pyro.get_param_store()
            n_clusters=self.level_sizes[-1]
            level_edges=[hardmax(pstore['edges_'+str(i)],axis=-1) for i in range(len(self.level_sizes)-1)]
            levels=self.tree_convergence_bottom_up.just_propagate(torch.eye(self.level_sizes[-1],device=pstore['locs'].device),level_edges)
            prop_taxon=torch.tensor(torch.cat(levels,dim=-1),device=pstore['locs'].device).float()
            
            discov_labels=adata.obs[self.discov_key].cat.categories
            latent_labels=[str(x) for x in range(pstore['discov_dc'].shape[1])]
            adata.obs[leaf_level]=adata.obs[leaf_level].astype('category')
            cluster_index=adata.obs[leaf_level].cat.categories.astype(int)#list(range(antipode_model.level_sizes[-1]))#list(range(pstore['locs'].shape[0]))
            cluster_labels=list(adata.obs[leaf_level].cat.categories)
            cluster_label_dict=dict(zip(cluster_index,cluster_labels))
            var_labels=adata.var.index
            
            prop_locs=prop_taxon@pstore['locs']
            prop_cluster_intercept=prop_taxon@pstore['cluster_intercept']
            cluster_params=((prop_locs@pstore['z_decoder_weight'])+prop_cluster_intercept+torch.mean(pstore['discov_constitutive_de'],0,keepdims=True))
            cluster_params=cluster_params[cluster_index,:]
            
            #Need to propagate multilayer tree to discovs
            prop_discov_di = torch.einsum('pc,dcg->dpg',prop_taxon,pstore['discov_di'])
            prop_discov_dm = torch.einsum('pc,dcm->dpm',prop_taxon,pstore['discov_dm'])
            discov_cluster_params=(torch.einsum('dpm,dmg->dpg',prop_locs+prop_discov_dm,pstore['z_decoder_weight']+pstore['discov_dc'])+(prop_cluster_intercept+prop_discov_di+pstore['discov_constitutive_de'].unsqueeze(1)))-pstore['softmax_shift']
            return torch.tensor(discov_cluster_params),torch.tensor(cluster_params), cluster_labels,var_labels,(torch.tensor(prop_taxon), torch.tensor(prop_locs),torch.tensor(prop_discov_di),torch.tensor(prop_discov_dm))
        else:
            pstore=adata.uns[prefix+'param_store']
            n_clusters=self.level_sizes[-1]
            level_edges=[numpy_hardmax(self.adata_manager.adata.uns[prefix+'param_store']['edges_'+str(i)],axis=-1) for i in range(len(self.level_sizes)-1)]
            levels=self.tree_convergence_bottom_up.just_propagate(np.eye(self.level_sizes[-1]),level_edges)
            prop_taxon=np.concatenate(levels,axis=-1)
            
            discov_labels=adata.obs[self.discov_key].cat.categories
            latent_labels=[str(x) for x in range(pstore['discov_dc'].shape[1])]
            adata.obs[leaf_level]=adata.obs[leaf_level].astype('category')
            cluster_index=adata.obs[leaf_level].cat.categories.astype(int)#list(range(antipode_model.level_sizes[-1]))#list(range(pstore['locs'].shape[0]))
            cluster_labels=list(adata.obs[leaf_level].cat.categories)
            cluster_label_dict=dict(zip(cluster_index,cluster_labels))
            var_labels=adata.var.index
            
            prop_locs=prop_taxon@pstore['locs']
            prop_cluster_intercept=prop_taxon@pstore['cluster_intercept']
            cluster_params=((prop_locs@pstore['z_decoder_weight'])+prop_cluster_intercept+np.mean(pstore['discov_constitutive_de'],0,keepdims=True))
            cluster_params=cluster_params[cluster_index,:]
            
            #Need to propagate multilayer tree to discovs
            prop_discov_di = np.einsum('pc,dcg->dpg',prop_taxon,pstore['discov_di'])
            prop_discov_dm = np.einsum('pc,dcm->dpm',prop_taxon,pstore['discov_dm'])
            discov_cluster_params=(np.einsum('dpm,dmg->dpg',prop_locs+prop_discov_dm,pstore['z_decoder_weight']+pstore['discov_dc'])+(prop_cluster_intercept+prop_discov_di+np.expand_dims(pstore['discov_constitutive_de'],1)))-pstore['softmax_shift']
            zero_mask = (adata.obs.groupby(self.discov_key)[leaf_level].value_counts().unstack().loc[:,cluster_labels]>=cluster_count_threshold).to_numpy()
            return discov_cluster_params,cluster_params, cluster_labels,var_labels, 1/zero_mask[...,np.newaxis],(prop_taxon, prop_locs,prop_discov_di,prop_discov_dm)
        
    def get_posterior_cluster_means(self, batch_size=128, device='cpu'):
        """
        Run data through the guide+model, aggregate per (discov, leaf) cluster,
        and return the posterior means.
        """
        # prepare obs categories
        leaf_level_col = 'level_' + str(len(self.level_sizes) - 1)
        adata = self.adata_manager.adata
        adata.obs[leaf_level_col] = pandas_numericategorical(
            adata.obs[leaf_level_col].astype(int)
        )
        # device & mode
        self.to(device)
        self.eval()
        self.freeze_encoder = False
        # build a quick loader to see if ind_x is present
        loader = scvi.dataloaders.AnnDataLoader(
            self.adata_manager,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            data_and_attributes=self.field_types
        )
        first = next(iter(loader))
        has_indices = ("ind_x" in first)
    
        # re‐init real loader
        loader = scvi.dataloaders.AnnDataLoader(
            self.adata_manager,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            data_and_attributes=self.field_types
        )
    
        # category info
        discov_cats = adata.obs[self.discov_key].cat.categories
        leaf_cats   = adata.obs[leaf_level_col].cat.categories.astype(int)
        num_discov  = len(discov_cats)
        num_leaf    = len(leaf_cats)
    
        # accumulators
        first_batch = True
        cell_counter = 0
    
        with torch.no_grad():
            for batch in tqdm.tqdm(loader):
                # move to device
                x = {k: v.to(device) for k, v in batch.items()}
                # trace guide→model
                guide_tr = poutine.trace(self.guide).get_trace(**x)
                model_tr = poutine.trace(poutine.replay(self.model, trace=guide_tr)) \
                                   .get_trace(**x)
                out = model_tr.nodes["_RETURN"]["value"]
                # from log‐rates to rates, numpy
                rates = torch.exp(out).cpu().numpy()
    
                if first_batch:
                    F = rates.shape[1]
                    sum_post   = np.zeros((num_discov, num_leaf, F), dtype=np.float64)
                    count_post = np.zeros((num_discov, num_leaf),    dtype=np.float64)
                    first_batch = False
    
                # compute which cells these are
                if has_indices:
                    idx = batch["ind_x"].cpu().numpy()
                else:
                    bsz = rates.shape[0]
                    idx = np.arange(cell_counter, cell_counter + bsz)
                    cell_counter += bsz
    
                # map to cluster codes
                d_codes = adata.obs[self.discov_key].cat.codes[idx].values
                l_codes = adata.obs[leaf_level_col].cat.codes[idx].values
    
                # accumulate
                for i, (dc, lc) in enumerate(zip(d_codes, l_codes)):
                    sum_post[dc, lc] += rates[i]
                    count_post[dc, lc] += 1
    
        # avoid divide‐by‐zero
        denom = np.maximum(count_post[..., None], 1)
        posterior_means = sum_post / denom
    
        return posterior_means

    def correct_fits(
        self,
        batch_size: int   = 128,
        device: str       = 'cpu',
        n_steps: int      = 100000,
        num_particles: int = 3,
    ):
        """
        1) Compute posterior cluster means via get_posterior_cluster_means
        2) Compute actual cluster means & orders
        3) Do the log‐residual + sub‐SVI step exactly as before
        """
        leaf_level_col = f'level_{len(self.level_sizes) - 1}'
        adata = self.adata_manager.adata
        adata.obs[leaf_level_col] = pandas_numericategorical(
            adata.obs[leaf_level_col].astype(int)
        )
    
        device = pyro.param("locs").device
        self.freeze_encoder = False
        self.to(device)
        self.eval()
    
        leaf_cats = adata.obs[leaf_level_col].cat.categories.astype(int)
        self.obs_leaves = leaf_cats
    
        posterior_means = self.get_posterior_cluster_means(
            batch_size=batch_size,
            device=device,
        )
    
        actual_means, actual_orders = group_aggr_anndata(
            adata,
            [self.discov_key, leaf_level_col],
            layer=self.layer,
            normalize=True,
        )
    
        discov_cluster_params, cluster_params, cluster_labels, var_labels, (
            prop_taxon, prop_locs, prop_discov_di, prop_discov_dm
        ) = self.calculate_cluster_params(flavor='torch')
        # select only the leaves
        discov_cluster_params = discov_cluster_params[:, leaf_cats, :]
    
        log_actual_means    = safe_log_transform(actual_means)
        min_val             = log_actual_means.min()
        log_posterior_means = np.log(posterior_means + np.exp(min_val))
        log_residuals_np    = log_actual_means - log_posterior_means
    
        log_residuals    = torch.tensor(log_residuals_np, dtype=torch.float32, device=device)
        corrected_means  = torch.log(
            torch.exp(discov_cluster_params + log_residuals) + np.exp(min_val)
        )
    
        non_leaf_offset = np.sum(self.level_sizes[:-1])
        leaf_cat_to_idx = {
            cat: (cat + non_leaf_offset)
            for cat in leaf_cats
        }
    
        for name, param in pyro.get_param_store().items():
            pyro.get_param_store()[name] = param.to(device)
        self.to(device)
    
        self.run_sub_svi_for_log_residuals(
            log_residuals   = corrected_means,
            leaf_cat_to_idx = leaf_cat_to_idx,
            sigma           = 1.0,
            lr              = 1e-2,
            n_steps         = n_steps,
            num_particles   = num_particles,
        )
    
        return posterior_means, actual_orders

    def sub_model_log_residuals(
        self,
        log_residuals: torch.Tensor,
        leaf_cat_to_idx: dict,
        sigma: float = 0.2,
    ):
        """
        A 'sub-model' that replicates the same final "discov_cluster_params" 
        calculation your snippet showed, then places a Normal likelihood on the 
        log-residuals. The shape of 'discov_cluster_params' is (num_discov, num_leaf, num_genes),
        matching the shape of log_residuals.
    
        Args:
          log_residuals: shape (num_discov, num_leaf, num_genes), i.e. the difference
                         [log_actual_means - log_posterior_means].
          antipode_model: your main model (self in correct_fits).
          leaf_cat_to_idx: map from each leaf category to the correct index 
                           in discov_di & cluster_intercept if needed (unused here).
          sigma: stdev for the Normal distribution that ties residuals together.
        """
    
        pyro.module("antipode_mini", self)
    
        # For convenience, gather the Pyro param store
        pstore = pyro.get_param_store()
    
        adata = self.adata_manager.adata
        device = log_residuals.device
    
        n_clusters = self.level_sizes[-1]
        level_edges = []
        for i in range(len(self.level_sizes) - 1):
            edges_i = pstore[f"edges_{i}"]  # e.g. shape: (some_dim, some_dim)
            edges_i_hard = hardmax(edges_i, axis=-1)
            level_edges.append(edges_i_hard)
    
        eye_bottom = torch.eye(n_clusters, device=device)
        levels = self.tree_convergence_bottom_up.just_propagate(eye_bottom, level_edges)
        prop_taxon = torch.cat(levels, dim=-1)  # shape: (n_clusters, sum_of_levels)
        
        locs = pstore["locs"].detach()                     # shape: (sum_of_levels, num_latent) fix locs so it remains compatible with dl inference
        # z_decoder_weight = pstore["z_decoder_weight"]      # shape: (num_latent, num_genes)
        # cluster_intercept = pstore["cluster_intercept"]    # shape: (sum_of_levels, num_genes)
        # discov_dc = pstore["discov_dc"]                    # shape: (num_discov, num_latent, num_genes)
        # discov_di = pstore["discov_di"]                    # shape: (num_discov, sum_of_levels, num_genes)
        # discov_dm = pstore["discov_dm"]                    # shape: (num_discov, sum_of_levels, num_latent)
        discov_constitutive_de = pstore["discov_constitutive_de"].detach()  # shape: (num_discov, num_genes)
        softmax_shift = pstore["softmax_shift"].detach()            # shape: scalar or (1,)
        
        # locs=self.zl.model_sample(log_residuals)
        discov_dm=self.dm.model_sample(log_residuals)
        discov_di=self.di.model_sample(log_residuals)
        cluster_intercept=self.ci.model_sample(log_residuals)
        z_decoder_weight=self.zdw.model_sample(log_residuals)
        discov_dc=self.dc.model_sample(log_residuals)
    
        prop_locs = prop_taxon @ locs
        # shape => (n_clusters, num_latent)
    
        prop_cluster_intercept = prop_taxon @ cluster_intercept
        # shape => (n_clusters, num_genes)
    
        cluster_params = (prop_locs @ z_decoder_weight) + prop_cluster_intercept + torch.mean(
            discov_constitutive_de, dim=0, keepdim=True
        )
        prop_discov_di = torch.einsum("pc, dcg -> dpg", prop_taxon, discov_di)
        prop_discov_dm = torch.einsum("pc, dcm -> dpm", prop_taxon, discov_dm)
    
        prop_locs_expanded = prop_locs.unsqueeze(0)  # (1, n_clusters, num_latent)
        prop_locs_plus_dm = prop_locs_expanded + prop_discov_dm
        # shape => (num_discov, n_clusters, num_latent)
    
        # z_decoder_weight + discov_dc => shape => (num_latent, num_genes) + (num_discov, num_latent, num_genes)
        # we need to broadcast across discov dimension => do an unsqueeze on z_decoder_weight for discov
        zdw_expanded = z_decoder_weight.unsqueeze(0)  # shape => (1, num_latent, num_genes)
        zdw_plus_discov_dc = zdw_expanded + discov_dc  # => shape => (num_discov, num_latent, num_genes)
    
        discov_cluster_expr = torch.einsum("dpm, dmg -> dpg", prop_locs_plus_dm, zdw_plus_discov_dc)
        pci_expanded = prop_cluster_intercept.unsqueeze(0)  # => (1, n_clusters, num_genes)
        di_plus_ci = pci_expanded + prop_discov_di  # shape => (num_discov, n_clusters, num_genes)
    
        # discov_constitutive_de => shape (num_discov, num_genes)
        # unsqueeze(1) => (num_discov, 1, num_genes) => broadcast
        dcd_unsqueezed = discov_constitutive_de.unsqueeze(1)
    
        discov_cluster_params = (
            discov_cluster_expr + di_plus_ci + dcd_unsqueezed
        ) - softmax_shift  # shape => (num_discov, n_clusters, num_genes)

        discov_cluster_params = discov_cluster_params[:,self.obs_leaves,:]

        num_discov, num_leaf, num_genes = log_residuals.shape
        with pyro.plate("disc", num_discov, dim=-3):
            with pyro.plate("leaf", num_leaf, dim=-2):
                with pyro.plate("gene", num_genes, dim=-1):
                    pyro.sample(
                        "residual_obs",
                        dist.Laplace(discov_cluster_params, sigma),
                        obs=log_residuals,
                    )
    
    def sub_guide_log_residuals(
        self,
        log_residuals: torch.Tensor,
        leaf_cat_to_idx: dict,
        sigma: float = 0.2,
    ):
        pyro.module("antipode_mini", self)
        # locs=self.zl.guide_sample(log_residuals)
        discov_dm=self.dm.guide_sample(log_residuals)
        discov_di=self.di.guide_sample(log_residuals)
        cluster_intercept=self.ci.guide_sample(log_residuals)
        z_decoder_weight=self.zdw.guide_sample(log_residuals)
        discov_dc=self.dc.guide_sample(log_residuals)
    
    def run_sub_svi_for_log_residuals(
        self,
        log_residuals: torch.Tensor,
        leaf_cat_to_idx: dict,
        sigma: float = 0.2,
        lr: float = 1e-4,
        n_steps: int = 5000,
        num_particles = 3
    ):
        """
        Runs a short SVI pass on the sub_model_log_residuals/sub_guide_log_residuals,
        effectively adjusting the MAPLaplaceModule parameters so that
        predicted offsets match the observed log_residuals.
        """
        
        optimizer = pyro.optim.ClippedAdam({"lr": lr})
        svi = SafeSVI(
            self.sub_model_log_residuals,
            self.sub_guide_log_residuals,
            optimizer,
            loss=pyro.infer.Trace_ELBO(num_particles=num_particles)
        )
        self.to(log_residuals.device)
        for step in range(n_steps):
            loss = svi.step(
                log_residuals, 
                leaf_cat_to_idx,
                sigma
            )
            if step % 1000 == 0:
                print(f"[sub-SVI step {step}] loss = {loss:.4f}")

    def correct_fits_intercepts(self, batch_size=128):
        """
        1) Shift all of the intercept‐type params to center them
        2) Compute posterior cluster means via get_posterior_cluster_means
        3) Compute log‐differences and apply to discov_di
        """
        # ————————————————————————————————————————————
        # 0) Prep leaf‐level column & save pre‐shift params
        leaf_level_col = f'level_{len(self.level_sizes)-1}'
        adata = self.adata_manager.adata
        adata.obs[leaf_level_col] = pandas_numericategorical(
            adata.obs[leaf_level_col].astype(int)
        )
        self.save_params_to_uns(prefix='precorrect_')
    
        # ————————————————————————————————————————————
        # 1) Shift pyro params so that discov_dm, discov_di, discov_dc are zero‐centered
        loc_shift = pyro.param('discov_dm').mean(0)
        # shift locs & dm
        pyro.get_param_store()['locs'] = (pyro.param('locs') + loc_shift).detach().clone()
        pyro.get_param_store()['discov_dm'] = (pyro.param('discov_dm') - loc_shift).detach().clone()
        # shift cluster_intercept & di
        di_shift = pyro.param('discov_di').mean(0)
        pyro.get_param_store()['cluster_intercept'] = (
            pyro.param('cluster_intercept') + di_shift
        ).detach().clone()
        pyro.get_param_store()['discov_di'] = (
            pyro.param('discov_di') - di_shift
        ).detach().clone()
        # shift z_decoder_weight & dc
        dc_shift = pyro.param('discov_dc').mean(0)
        pyro.get_param_store()['z_decoder_weight'] = (
            pyro.param('z_decoder_weight') + dc_shift
        ).detach().clone()
        pyro.get_param_store()['discov_dc'] = (
            pyro.param('discov_dc') - dc_shift
        ).detach().clone()
    
        # save post‐shift params
        self.save_params_to_uns()
    
        # ————————————————————————————————————————————
        # 2) Prep model & device
        device = pyro.param('locs').device
        self.freeze_encoder = False
        self.to(device)
        self.eval()
    
        # pull out category arrays for later
        discov_cats = adata.obs[self.discov_key].cat.categories
        leaf_cats   = adata.obs[leaf_level_col].cat.categories
    
        # ————————————————————————————————————————————
        # 3) Compute posterior means via helper
        posterior_means = self.get_posterior_cluster_means(
            batch_size=batch_size,
            device=device,
        )  # shape (num_discov, num_leaf, num_features)
    
        # ————————————————————————————————————————————
        # 4) Compute actual cluster means & orders
        actual_aggr = group_aggr_anndata(
            adata,
            [self.discov_key, leaf_level_col],
            layer=self.layer,
            normalize=True,
        )
        sum_aggr = group_aggr_anndata(
            adata,
            [leaf_level_col],
            layer=self.layer,
            normalize=False,
            agg_func=np.sum,
        )
        actual_means  = actual_aggr[0]
        actual_orders = actual_aggr[1][leaf_level_col]
        # total counts per leaf → for log transform
        actual_sums = sum_aggr[0].sum(-1)[np.newaxis, ..., np.newaxis]
    
        # ————————————————————————————————————————————
        # 5) Log‐transform and compute delta for discov_di
        log_actual_means    = safe_log_transform(actual_means, actual_sums)
        log_posterior_means = safe_log_transform(posterior_means)
        add_to_discov_di    = log_actual_means - log_posterior_means
    
        # ————————————————————————————————————————————
        # 6) Scatter into full discov_di array
        current_di = pyro.param('discov_di')
        non_leaf  = np.sum(self.level_sizes[:-1])
        # build map leaf_cat → index in full discov_di
        leaf_cat_to_idx = {cat: (cat + non_leaf) for cat in leaf_cats}
    
        # final delta array
        final_delta = np.zeros_like(current_di.detach().cpu().numpy())
        for i, cat in enumerate(leaf_cats):
            idx = leaf_cat_to_idx[cat]
            final_delta[:, idx, :] = add_to_discov_di[:, i, :]
    
        # sanity check
        if final_delta.shape != current_di.shape:
            raise ValueError(
                f"Shape mismatch: discov_di {current_di.shape} vs delta {final_delta.shape}"
            )
    
        # ————————————————————————————————————————————
        # 7) Write back updated discov_di
        with torch.no_grad():
            new_di = (current_di + torch.tensor(final_delta, device=device)).float()
            new_di.requires_grad_()
            pyro.get_param_store()['discov_di'] = new_di
    
        # 8) Return the core results
        posterior_orders = {
            leaf_level_col: leaf_cats,
            self.discov_key: discov_cats,
        }
        return posterior_means, posterior_orders


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
                try:
                    pyro.get_param_store().__delitem__(k)
                except:
                    print(k,'not deleted')
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
            try:
                self.adata_manager.adata.var.drop('_index',axis=1,inplace=True)
            except:
                pass
            try:
                self.adata_manager.adata.obs.drop('_index',axis=1,inplace=True)
            except:
                pass
            try:
                self.adata_manager.adata.raw.var.drop('_index',axis=1,inplace=True)
            except:
                pass            
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
        device = 'cpu',
        load_kw_args = {'backed':'r'}
    ) -> tuple[dict, np.ndarray, dict, AnnData]:
        """Helper to load saved files."""
        file_name_prefix = prefix or ""
    
        model_file_name = f"{file_name_prefix}model.pt"
        model_path = os.path.join(dir_path, model_file_name)
        try:
            model = torch.load(model_path,map_location=device,weights_only=False)# used to be default weights_only=False
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
            device=device,
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

