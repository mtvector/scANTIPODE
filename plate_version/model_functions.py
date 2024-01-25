import torch
import torch.nn as nn
from torch.distributions import constraints
import numpy as np
import tqdm
import pyro
import scipy
from torch.nn.functional import softplus, softmax
from torch.optim import Adam
import seaborn
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.ops.indexing import Vindex
import scanpy as sc
import math
import scvi

def null_function(x):
    return x

def safe_sigmoid(x,eps=1e-10):
    #return torch.clamp(torch.sigmoid(x),min=eps,max=(1.-eps))
    return (torch.sigmoid(x)+1e-6)*(1-1e-5)

def centered_sigmoid(x):
    return (2*(torch.sigmoid(x)-0.5))

def safe_softmax(x,dim=-1):
    x=torch.softmax(x,dim)
    x=x+1e-10
    return (x/x.sum(dim,keepdim=True))

def minmax(x):
    return(x.min(),x.max())

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def csr_to_sparsetensor(x):
    '''
    Convert scipy csr sparse matrix to sparse tensor
    '''
    coo=x.tocoo()
    return(torch.sparse.LongTensor(torch.LongTensor([coo.row.tolist(), coo.col.tolist()]),
                              torch.Tensor(coo.data.astype(np.float64))))
 

def param_store_to_numpy():
    store={}
    for name in pyro.get_param_store():
        store[name]=pyro.param(name).cpu().detach().numpy()
    return store

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
            if num_outs==1:
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
    Take a tensor of inputs

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
            if num_outs==1:
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
    if 'species_onehot' not in antipode_model.adata_manager.adata.obsm.keys():
        antipode_model.adata_manager.adata.obsm['species_onehot']=numpy_onehot(antipode_model.adata_manager.adata.obs['species'].cat.codes)
    antipode_model.adata_manager.register_new_fields([scvi.data.fields.ObsmField('species_onehot','species_onehot')])

    field_types={"s":np.float32,"species_onehot":np.float32}
    dataloader=scvi.dataloaders.AnnDataLoader(antipode_model.adata_manager,batch_size=32,drop_last=False,shuffle=False,data_and_attributes=field_types)#supervised_field_types for supervised step 
    encoder_outs=batch_output_from_dataloader(dataloader,antipode_model.zl_encoder,batch_size=batch_size,device=device)
    encoder_outs[0]=antipode_model.z_transform(encoder_outs[0])
    encoder_out=[x.detach().cpu().numpy() for x in encoder_outs]
    classifier_outs=batch_torch_outputs([(antipode_model.z_transform(encoder_outs[0]))],antipode_model.classifier,batch_size=2048,device='cuda')
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

def fest(tensors,unsqueeze=0,epsilon=1e-10):
    '''
    flexible_einsum_scale_tensor, first dimension must be equal for list of tensors
    Multiplies out marginals to construct joint
    '''
    einsum_str = ','.join(f'...z{chr(65 + i)}' for i, _ in enumerate(tensors))
    einsum_str += '->...' + ''.join(chr(65 + i) for i, _ in enumerate(tensors))
    out=torch.einsum(einsum_str, *[x/(x.sum(-1,keepdim=True)) for x in tensors])[*indexing_none_list(unsqueeze)]
    #print(out.shape)
    return [poutine.scale(scale=out+epsilon)]

class ZLEncoder(nn.Module):
    '''
    Takes tensor of size (batch,num_var) input (for now)

        Args:
        num_var (integer): size of input variables (e.g. number of genes)
        outputs ([(),()]): example [(self.z_dim,None),(self.z_dim,softplus),(1,None),(1,softplus),(1,None),(1,softplus)]
        hidden_dims ([integer]) 
    '''
    def __init__(self, num_var, outputs, hidden_dims=[],num_cat_input=0,hidden_conv_channels=4,conv_out_channels=1):
        super().__init__()
        self.num_cat_input=num_cat_input
        self.output_dim=[x[0] for x in outputs] 
        self.output_transform=[x[1] for x in outputs]
        self.cumsums=np.cumsum(self.output_dim)
        self.cumsums=np.insert(self.cumsums,0,0)
        self.dims = [conv_out_channels*num_var+self.num_cat_input+1] + hidden_dims + [self.cumsums[-1]]
        self.fc = make_fc(self.dims,dropout=True)

    def forward(self, s,species=None):
        # Transform the counts x to log space for increased numerical stability.
        # Note that we only use this transformation here; in particular the observation
        # distribution in the model is a proper count distribution.
        s_sum=torch.log(1 + s.sum(-1).unsqueeze(-1))
        s = torch.log(1 + s)
        if species is None:
            x=self.fc(torch.cat([s,s_sum],dim=-1))
        else:
            x=self.fc(torch.cat([s,species,s_sum],dim=-1))
        return_list=[]
        for i in range(len(self.cumsums)-1):
            if self.output_transform[i] is None:
                return_list.append(x[:,self.cumsums[i]:self.cumsums[i+1]])
            else:
                return_list.append(self.output_transform[i](x[:,self.cumsums[i]:self.cumsums[i+1]]))
        return(return_list)    
    
class ZDecoder(nn.Module):
    '''
    Neural network has input dimension num_latent and output dimension num_var.
    '''
    def __init__(self, num_latent ,num_var, hidden_dims=[]):
        super().__init__()
        dims = [num_latent] + hidden_dims + [num_var]
        #self.fc = make_fc(dims)
        #self.bn=nn.BatchNorm1d(num_var)
        
    def forward(self,z,weight,delta=None):
        if delta is None:
            mu=torch.einsum('bi,ij->bj',z,weight)
        else:
            mu=torch.einsum('bi,bij->bj',z,weight+delta)
        #mu=self.bn(mu)
        return mu     


class Classifier(nn.Module):
    '''
    Simple FFNN with output splitting
    '''
    def __init__(self, num_latent, outputs,hidden_dims=[2000,2000,2000]):
        super().__init__()
        self.output_dim=[x[0] for x in outputs] 
        self.output_transform=[x[1] for x in outputs]
        self.cumsums=np.cumsum(self.output_dim)
        self.cumsums=np.insert(self.cumsums,0,0)
        self.dims = [num_latent] + hidden_dims + [self.cumsums[-1]]
        self.fc = make_fc(self.dims,dropout=False)

    def forward(self, z):
        x=self.fc(z)
        return_list=[]
        for i in range(len(self.cumsums)-1):
            if self.output_transform[i] is None:
                return_list.append(x[:,self.cumsums[i]:self.cumsums[i+1]])
            else:
                return_list.append(self.output_transform[i](x[:,self.cumsums[i]:self.cumsums[i+1]]))
        return(return_list)   

    
class DMCorrectOutput(nn.Module):
    '''
    Subtracts/adds back DM from latent space.(shouldn't be necessary unless using archaic model)
    '''
    def __init__(self, species_dm,batch_dm):
        super().__init__()
        self.species_dm=species_dm
        self.batch_dm=batch_dm
        
    def forward(self, x,o2,batch_values,species_values):
        return((x-
                 torch.einsum('bi,bij->bj',o2,
                              pyro.param('batch_dm')[batch_values.argmax(1),...])-
                 torch.einsum('bi,bij->bj',o2,
                              pyro.param('species_dm')[species_values.argmax(1),...])))          
    
class SimpleFFNN(nn.Module):
    '''
    Basic feed forward neural network
    '''
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        self.fc = make_fc(dims,dropout=True)

    def forward(self, x):
        return self.fc(x)
    
def mixture(x,y,psi):
    return((psi*x)+((1-psi)*y))

class TGeN(nn.Module):
    '''
    Trajectory Generator Network
    '''
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.linear1=nn.Linear(in_dim, hidden_dim,bias=True)
        self.relu=nn.ReLU()
        self.bn1=torch.nn.BatchNorm1d(hidden_dim)
        self.bn2=torch.nn.BatchNorm1d(hidden_dim)
        #self.bn3=torch.nn.BatchNorm1d(hidden_dim)
        self.conv1=nn.Conv1d(hidden_dim,hidden_dim,kernel_size=(1,), stride=(1,),bias=True,groups=1)
        self.conv2=nn.Conv1d(hidden_dim,hidden_dim,kernel_size=(1,), stride=(1,),bias=True,groups=1)
        #self.conv3=nn.Conv1d(hidden_dim,hidden_dim,kernel_size=(1,), stride=(1,),bias=True,groups=hidden_dim)

    def forward(self, x,out_weights=None):
        x=self.relu(self.bn1(self.linear1(x)))
        x=self.relu(self.conv1(x.unsqueeze(-1)).squeeze())#
        x=self.bn2(self.conv2(x.unsqueeze(-1)).squeeze())
        #x=self.bn3(self.conv3(x.unsqueeze(-1)).squeeze())
        return x

def softplus_sum(z):
    '''
    Transforms to simplex in linear space rather than softplus' exponential
    '''
    z=torch.nn.functional.relu(z)
    z=z+1e-8
    z=z/z.sum(-1).reshape(-1,1)
    return(z)

def exp_sum(a,b):
    '''
    Add two exponentials (in linear space)
    '''
    return((a.exp()+b.exp()).log())

def index_to_onehot(index, out_shape):
    if sum(index.shape) == 1:
        index=torch.zeros(out_shape)
    else:
        index=torch.nn.functional.one_hot(index.squeeze(),num_classes=out_shape[1]).float() if index.shape[-1]==1 else index
    return index

def numpy_onehot(x,num_classes=None):
    n_values = np.max(x) + 1
    if num_classes is None or num_classes<n_values:
        num_classes=n_values
    return np.eye(num_classes)[x]

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

def add_cats_uns(adata,column,uns_name=None):
    if uns_name is None:
        uns_name=column+'_cats'
    adata.uns[uns_name]=dict(zip([str(x) for x in adata.obs[column].cat.categories],[str(x) for x in sorted(set(adata.obs[column].cat.codes))]))

def gen_exponential_decay(a):
    def exponential_decay(x, k):
        return k - (k - 1) * torch.exp(-a * x)
    return exponential_decay  # Corrected to return the inner function

def gen_linear_function(n, start_point):
    def linear_function(x, k):
        if x < start_point:
            return 1
        else:
            return 1 + ((k - 1) / (n - start_point)) * (x - start_point)
    return linear_function

def make_dataloader(origdata=None,adata_path=None,batch_size=32):
    '''
    Loads anndata chunks (specific for development dataset)
    TODO make general dataloader
    TODO implement sparse tensor dataloader?
    '''
    import gc    
    if adata_path is not None:
        adata=sc.read_h5ad(adata_path)
    else:
        print('Provide a path')
    if origdata is not None:
        adata=adata[:,adata.var.index.isin(origdata.var.index)]
    
    #adata.obs['species']=adata.obs['species'].astype('category')
    species_arg=adata.obs['species'].cat.codes
    species_values=torch.nn.functional.one_hot(torch.tensor(adata.obs['species'].cat.codes).long(),num_classes=len(adata.obs['species'].cat.categories)).float()

    #adata.obs['batch_name']=adata.obs['batch_name'].astype('category')
    batch_arg=adata.obs['batch_name'].replace(adata.uns['batch_cats']).astype(int)
    batch_values=torch.nn.functional.one_hot(torch.tensor(adata.obs['batch_name'].replace(adata.uns['batch_cats']).astype(int)).long(),
                                             num_classes=len(adata.uns['batch_cats'].keys())).float()
    
    #adata.obs['region_species']=adata.obs['sample_region'].astype(str)+'_'+adata.obs['species'].astype(str)
    class_weights = adata.obs['region_species'].astype('category').cat.codes.value_counts(normalize=False,sort=False).sort_index()
    labels = adata.obs['region_species'].astype('category').cat.codes #corresponding labels of samples
    weights = [1/class_weights[labels[i]] for i in range(len(labels))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights),len(weights),replacement=True)

    spliced_counts=adata.layers['spliced']
    del adata
    gc.collect()
    d=torch.utils.data.TensorDataset(torch.tensor(spliced_counts.todense()),species_values,batch_values)
    del spliced_counts
    gc.collect()
    dataloader = torch.utils.data.DataLoader(d,sampler=sampler, batch_size=batch_size,drop_last=True)
    return(dataloader)

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

def split_in_half(t):
    '''
    Splits a tensor in half along the final dimension from tutorial
    '''
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)

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

class SUConvModule(nn.Module):
    '''
    Original SU combo module maintains information within channel
    '''
    def __init__(self, num_var,hidden_channels=4,out_channels=2):
        super().__init__()
        self.conv=torch.nn.Conv1d(2,hidden_channels,kernel_size=(1,), stride=(1,),bias=True)
        self.relu=torch.nn.ReLU()
        self.conv2=torch.nn.Conv1d(hidden_channels,out_channels,kernel_size=(1,), stride=(1,),bias=False)
        self.bn1=torch.nn.LayerNorm(num_var,hidden_channels)
        self.bn2=torch.nn.BatchNorm1d(out_channels*num_var)

    def forward(self, s,u):
        x=self.conv(torch.stack([s,u],dim=1))#self.bn1(
        x=self.relu(x)
        x=self.bn2(torch.flatten(self.conv2(x),start_dim=1,end_dim=-1))
        return(x)
    
class SUTransConvModule(nn.Module):
    '''
    Original SU transpose conv module maintains information within channel
    '''
    def __init__(self, num_var,hidden_channels=4,out_channels=2):
        super().__init__()
        self.conv=torch.nn.ConvTranspose1d(1,hidden_channels,kernel_size=(1,), stride=(1,),bias=True)
        self.relu=torch.nn.ReLU()
        self.conv2=torch.nn.ConvTranspose1d(hidden_channels,out_channels,kernel_size=(1,), stride=(1,),bias=False)
        self.bn1=torch.nn.LayerNorm(num_var,hidden_channels)
        self.bn2=torch.nn.BatchNorm1d(out_channels*num_var)

    def forward(self, x):
        x=self.conv(x.unsqueeze(1))#self.bn1(
        x=self.relu(x)
        x=self.conv2(x)
        s,u=torch.unbind(x,dim=-2)
        return(s,u)
