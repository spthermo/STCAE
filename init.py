from encoder import *
from seq2seg import *

def get_model(name, model_params):
    if name == 'default':
        return Seq2Seg(model_params['dim'], model_params['ndf'], model_params['dilation'])
    else:
        print("Could not find the requested model ({})".format(name), file=sys.stderr)