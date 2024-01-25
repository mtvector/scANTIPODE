import scvi
import torch
import anndata
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

