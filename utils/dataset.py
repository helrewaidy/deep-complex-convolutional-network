from typing import Tuple, List
import numpy as np
from torch.utils.data import Dataset, DataLoader
from configs import config


def create_cmplx_dataset(data_shape: Tuple) -> np.ndarray:
    """Create a dataset of complex images.
    
    Parameters
    ----------
    data_shape : Tuple
        The shape of the data.
    
    Returns
    -------
    np.ndarray
        A dataset of complex random values.
    """
    return np.random.rand(*data_shape)


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """Get the dataset loaders.

    Returns
    -------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        The training and validation data loaders
    """
    input_shape = (config.in_channels, *config.input_shape)
    tr_dataset = DataGenerator(input_shape, list(range(80)))
    vld_dataset = DataGenerator(input_shape, list(range(20)))

    tr_data_loader = DataLoader(
        tr_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.data_loaders_num_workers
    )

    vld_data_loader = DataLoader(
        vld_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.data_loaders_num_workers
    )

    return tr_data_loader, vld_data_loader


class DataGenerator(Dataset):
    def __init__(self, input_shape: Tuple, data_indicies: List):
        """Initialize the DataGenerator class.

        Parameters
        ----------
        input_shape : Tuple
            The shape of the input data.
        """
        self.input_shape = input_shape
        self.data_indicies = data_indicies
    
    def __len__(self):
        return len(self.data_indicies)

    def __getitem__(self, index):
        'Generate one batch of data'
        usr = 4
        X = create_cmplx_dataset(self.input_shape)
        ds_shape = [s//usr for s in range(1, len(self.input_shape)-1)]
        y = np.resize(np.resize(X.copy(), ds_shape), self.input_shape)
        return X, y
