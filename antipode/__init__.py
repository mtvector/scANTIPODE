#from .antipode_model import ANTIPODE
#from .model_distributions import *
#from .model_functions import *
#from .model_modules import *
from antipode import model_distributions
from antipode import model_functions
from antipode import plotting
from antipode import model_modules
from antipode import train_utils
from antipode import post
from antipode import anndata_utils
from antipode import antipode_model
from antipode import phylo

# You might also define some basic information or helper functions here
__version__ = '0.1.0'
__author__ = 'Matthew Schmitz'

def get_version():
    return __version__
