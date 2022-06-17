

class Config(object):
    """Global Config class"""

    # Dataset configs
    spatial_dimentions = 2
    input_shape = (256, 256, 2)  # (x_dim, y_dim, z_dim, real-imag)
    in_channels = 1
    out_channels = 1
    batch_size = 2
    data_loaders_num_workers = 4

    # Training configs
    learning_rate = 0.01
    models_dir = 'models/'
    workspace_dir = 'workspace/'
    num_epochs = 50
    normalize_input = True

    # Model configs
    unet_global_residual_conn = False
    kernel_size = 3
    bias = False
    activation = 'CReLU'
    activation_params = {
        'inplace': True,
    }
    bn_t = 5
    dropout_ratio = 0.0


config = Config()


