import scvi
import torch
import anndata
import tqdm
import numpy as np
from IPython.display import Audio, display

def allDone():
    display(Audio(url='https://notification-sounds.com/soundsfiles/Meditation-bell-sound.mp3', autoplay=True))

def make_field(name,loc):
    if loc[0] == 'obsm':
        field=scvi.data.fields.ObsmField
    if loc[0] == 'obs':
        field=scvi.data.fields.CategoricalObsField
    if loc[0] == 'layers':
        field=scvi.data.fields.LayerField
    return(field(name,loc[1]))

def get_field(adata,loc):
    return adata.__getattribute__(loc[0]).__getattribute__(loc[1])

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

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

def numpy_onehot(x,num_classes=None):
    n_values = np.max(x) + 1
    if num_classes is None or num_classes<n_values:
        num_classes=n_values
    return np.eye(num_classes)[x]

def numpy_hardmax(x,axis=-1):
    return(numpy_onehot(x.argmax(axis).flatten(),num_classes=x.shape[axis]))

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

