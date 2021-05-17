import sys

from models.model import EncLSTM, Decoder

def get_model(name, model_params):
    if name == 'default':
        return EncLSTM(
                    model_params['dim'],
                    model_params['ndf'],
                    model_params['affordance_classes'],
                    model_params['ngroups'],
                    model_params['nchannels']
                ),\
                Decoder(
                    model_params['dim'],
                    model_params['ndf'],
                    model_params['affordance_classes'],
                    model_params['ngroups']
                )
    else:
        print("Could not find the requested model ({})".format(name), file=sys.stderr)