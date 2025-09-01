from .UNetLike import UNetLike, UNetLike2
from .ThermoPT import ThermoPT


registered_models= {
    'unet_like': UNetLike2,
    'thermo_pt': ThermoPT,
}

def get_registered_models(model_name, config):
    if 'unet_like' in model_name:
        in_channels = config['chunk_size']
        return registered_models[model_name](in_channels = in_channels, out_channels = config['out_channels'], init_features = config['init_features'])
    if 'thermo_pt' in model_name:
        enable_aov_refine = config.get('enable_aov_refine', True)
        enable_bev_refine = config.get('enable_bev_refine', True)
        enable_reconstruction = config.get('enable_reconstruction', True)
        return registered_models[model_name](exp_config = config, in_channels = config['chunk_size'], out_channels = config['out_channels'], init_features = config['init_features'], enable_aov_refine = enable_aov_refine, enable_bev_refine = enable_bev_refine, enable_reconstruction = enable_reconstruction)
    else:
        print("The model name is not registered!")
        return None