import scvi
import torch
import anndata
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softplus, softmax
import scipy
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import scanpy as sc
import math
from . import model_functions
from .model_functions import *

def make_field(name,loc):
    if loc[0] == 'obsm':
        field=scvi.data.fields.ObsmField
    if loc[0] == 'obs':
        field=scvi.data.fields.CategoricalObsField
    if loc[0] == 'layers':
        field=scvi.data.fields.LayerField
    return(field(name,loc[1]))


def csr_to_sparsetensor(x):
    '''
    Convert scipy csr sparse matrix to sparse tensor
    '''
    coo=x.tocoo()
    return(torch.sparse.LongTensor(torch.LongTensor([coo.row.tolist(), coo.col.tolist()]),
                              torch.Tensor(coo.data.astype(np.float64))))

def batch_torch_outputs(inputs,function,batch_size=2048,device='cuda'):
    '''
    Take a tensor of inputs

        Args:
        inputs ([Tensor]): List of input tensors to be batched along 0 dimension
        function (function(Tensor)): Torch module with forward() method implemented, like a classifier
        batch_size (integer): number along 0 dim per batch
        device (string): ('cuda','cpu',...)
    '''
    num_obs=inputs[0].shape[0]
    out_list=[[]]
    function.to(device)
    with torch.no_grad():
        for i in tqdm.tqdm(range(int(num_obs/batch_size)+1)):
            end_ind=min(((i+1)*batch_size),num_obs)
            if (i*batch_size) == end_ind:
                continue
            outs=function(*[x[(i*batch_size):end_ind].to(device) for x in inputs])
            num_outs=len(outs)
            if num_outs==1 or type(outs) is not list:
                num_outs=1
                out_list[0].append(outs.to('cpu'))
            else:
                for j in range(num_outs):
                    if j==len(out_list):
                        out_list.append([outs[j].to('cpu')])
                    else:
                        out_list[j].append(outs[j].to('cpu'))
        final_outs=[torch.cat(out_list[i],dim=0) for i in range(num_outs)]
        return(final_outs)    

#Make full dataloader first
def batch_output_from_dataloader(dataloader,function,batch_size=2048,device='cuda'):
    '''
    Take a dataloader and apply the function to all steps

        Args:
        dataloader ([AnnDataLoader]): AnnDataLoader that only returns what's needed for function (torch module)
        function (function(Tensor)): Torch module with forward() method implemented, like a classifier
        batch_size (integer): number along 0 dim per batch
        device (string): ('cuda','cpu',...)
    '''
    out_list=[[]]
    function.to(device)
    function.eval()
    with torch.no_grad():
        for x in tqdm.tqdm(dataloader):
            x=[x[k].to(device) for k in x.keys()]
            outs=function(*x)
            num_outs=len(outs)
            if num_outs==1 or type(outs) is not list:
                num_outs=1
                out_list[0].append(outs.to('cpu'))
            else:
                for j in range(num_outs):
                    if j==len(out_list):
                        out_list.append([outs[j].to('cpu')])
                    else:
                        out_list[j].append(outs[j].to('cpu'))

        final_outs=[torch.cat(out_list[i],dim=0) for i in range(num_outs)]
        return(final_outs)

def get_antipode_outputs(antipode_model,batch_size=2048,device='cuda'):
    design_matrix=False  #3x faster
    
    if antipode_model.discov_key not in antipode_model.adata_manager.adata.obsm.keys():
        onehot_key=antipode_model.discov_key+"_onehot"
        antipode_model.adata_manager.adata.obsm[onehot_key]=numpy_onehot( antipode_model.adata_manager.adata.obs[antipode_model.discov_key].cat.codes)
    else:
        onehot_key=antipode_model.discov_key
    antipode_model.adata_manager.register_new_fields([scvi.data.fields.ObsmField(onehot_key,onehot_key)])       
    field_types={"s":np.float32,onehot_key:np.float32}
    dataloader=scvi.dataloaders.AnnDataLoader(antipode_model.adata_manager,batch_size=32,drop_last=False,shuffle=False,data_and_attributes=field_types)#supervised_field_types for supervised step 
    encoder_outs=batch_output_from_dataloader(dataloader,antipode_model.zl_encoder,batch_size=batch_size,device=device)
    encoder_outs[0]=antipode_model.z_transform(encoder_outs[0])
    encoder_out=[x.detach().cpu().numpy() for x in encoder_outs]
    classifier_outs=batch_torch_outputs([encoder_outs[0]],antipode_model.classifier,batch_size=2048,device='cuda')
    classifier_out=[x.detach().cpu().numpy() for x in classifier_outs]
    return encoder_out,classifier_out

def indexing_none_list(n):
    '''create unsqueeze n times. Negative values go to the end of the list; positive the front (for fest)'''
    none_list = [...]
    if n == 0:
        return none_list
    abs_n = abs(n)

    for _ in range(abs_n):
        if n < 0:
            none_list.append(None)
        else:
            none_list.insert(0, None)
    return none_list

def fest(tensors,unsqueeze=0,scalar=1.,epsilon=1e-10):
    '''
    flexible_einsum_scale_tensor, first dimension must be equal for list of tensors
    Multiplies out marginals to construct joint
    '''
    einsum_str = ','.join(f'...z{chr(65 + i)}' for i, _ in enumerate(tensors))
    einsum_str += '->...' + ''.join(chr(65 + i) for i, _ in enumerate(tensors))
    out=torch.einsum(einsum_str, * [x/(x.sum(-1,keepdim=True)) for x in tensors])[*indexing_none_list(unsqueeze)]
    #print(out.shape)
    return [poutine.scale(scale=scalar*out+epsilon)]


def softplus_sum(z):
    '''Transforms to simplex in linear space rather than softplus' exponential'''
    z=torch.nn.functional.relu(z)
    z=z+1e-8
    z=z/z.sum(-1).reshape(-1,1)
    return(z)

def beta_parameters_from_mean_variance(mu, sigma_squared):
    '''Calculate beta distribution parameters given mu and sigma^2'''
    a = mu * (mu * (1 - mu) - sigma_squared) / sigma_squared
    b = a * (1 / mu - 1)
    return a, b

def index_to_onehot(index, out_shape):
    if sum(index.shape) == 1:
        index=torch.zeros(out_shape)
    else:
        index=torch.nn.functional.one_hot(index.squeeze(),num_classes=out_shape[1]).float() if index.shape[-1]==1 else index
    return index

def oh_index(mat,ind):
    '''
    treat onehot as categorical index for 2d input
    '''
    return(torch.einsum('...ij,...bi->...bj',mat,ind))

def oh_index1(mat,ind):
    '''
    treat onehot as categorical index for 3d input
    '''
    return(torch.einsum('...ijk,...bi->...bjk',mat,ind))

def oh_index2(mat,ind):
    '''
    treat onehot as categorical index for 3d input
    '''
    return(torch.einsum('...bij,...bi->...bj',mat,ind))

def np_oh_index1(mat,ind):
    '''
    treat onehot as categorical index for 3d input
    '''
    return(np.einsum('...ijk,...bi->...bjk',mat,ind))

def np_oh_index2(mat,ind):
    '''
    treat onehot as categorical index for 3d input
    '''
    return(np.einsum('...bij,...bi->...bj',mat,ind))


def add_cats_uns(adata,column,uns_name=None):
    if uns_name is None:
        uns_name=column+'_cats'
    adata.uns[uns_name]=dict(zip([str(x) for x in adata.obs[column].cat.categories],[str(x) for x in sorted(set(adata.obs[column].cat.codes))]))

def gen_exponential_decay(a):
    def exponential_decay(x, k):
        return k - (k - 1) * torch.exp(-a * x)
    return exponential_decay

def gen_linear_function(n, start_point):
    def linear_function(x, k):
        if x < start_point:
            return 1
        else:
            return 1 + ((k - 1) / (n - start_point)) * (x - start_point)
    return linear_function

def make_fc(dims,dropout=False):
    '''
    Helper for making fully-connected neural networks from tutorial
    '''
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim,bias=False))
        if dropout:
            layers.append(nn.Dropout(0.05))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers[:-1])  # Exclude final ReLU non-linearity

def enumeratable_bn(x,bn):
    '''
    Batch norm that can work with categorical enumeration, from scANVI tutorial
    '''
    if len(x.shape) > 2:
        _x = x.reshape(-1, x.size(-1))
        _x=bn(_x)
        x = _x.reshape(x.shape[:-1] + _x.shape[-1:])
    else:
        x=bn(x)
    return(x)

def stick_break(beta):
    '''
    Stick breaking process using Beta distributed values along the last dimension
    '''
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return torch.nn.functional.pad(beta, (0, 1), value=1) * torch.nn.functional.pad(beta1m_cumprod, (1, 0), value=1)

def init_kaiming_weight(wt):
    '''
    Initialize weights by kaiming uniform
    '''
    torch.nn.init.kaiming_uniform_(wt, a=math.sqrt(5))
    
def init_uniform_bias(bs,wt):
    '''
    Initialize biases by kaiming uniform
    '''
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(wt)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    torch.nn.init.uniform_(bs, -bound, bound)    

def create_weighted_random_sampler(series):
    # Count the occurrences of each category in the Series
    class_counts = series.value_counts()
    # Calculate the weight for each category (inverse of count)
    class_weights = 1. / class_counts
    # Map the weights to the original series to assign a weight to each item
    weights = series.map(class_weights)
    # Convert weights to a tensor
    sample_weights = torch.DoubleTensor(weights.values)
    # Create the WeightedRandomSampler
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

def mixture(x,y,psi):
    return((psi*x)+((1-psi)*y))

