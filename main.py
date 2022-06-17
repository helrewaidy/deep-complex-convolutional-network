from cmath import inf
import shutil
import os

import torch.nn.modules.loss as Loss
from torch import optim
import numpy as np

from complex_net.cmplx_unet import CUNet
from complex_net.cmplx_blocks import batch_norm
from utils.dataset import get_dataloaders
from utils.loss import SSIM
from configs import config
import logging
from complex_layers.radial_bn import RadialNorm


logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


def set_seeds(seed):
    """Set the seeds for reproducibility
    
    Parameters
    ----------
    seed : int
        The seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device():
    """Get device

    Returns
    -------
    device : torch.device
        The device to use.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(net, optimizer, loss_criterion, tr_dataloader, epoch):
    """Train for one epoch of the data

    Parameters
    ----------
    net : torch.nn.Module
        The network to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    loss_criterion : torch.nn.Module
        The loss criterion to use.
    tr_dataloader : torch.utils.data.DataLoader
        The training data loader.
    epoch : int
        The epoch number.

    Returns
    -------
    avg_loss : float
        The average loss for the epoch.
    net : torch.nn.Module
        The trained network.
    optimizer : torch.optim.Optimizer
        The optimizer.
    """
    avg_loss = 0.0
    net.train()
    device = get_device()
    radial_normalizer = batch_norm(
        in_channels =config.in_channels,
    )
    for itt, (input, target) in enumerate(tr_dataloader):
        X = Variable(torch.FloatTensor(input.float())).to(device)
        y = Variable(torch.FloatTensor(target.float())).to(device)

        if config.normalize_input:
            X = radial_normalizer(X)
            y = radial_normalizer(y)

        y_pred = net(X)

        loss = loss_criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.detach().item() / len(tr_dataloader)

        logging.info('Epoch: {0} - Itter: {1}/{2} - loss: {3:.6f}'.format(
            epoch, itt, len(tr_dataloader), loss.detach().item())
        )

    return avg_loss, net, optimizer


def validate(net, loss_criterion, val_dataloader, epoch):
    """Validate the model on the validation set

    Parameters
    ----------
    net : torch.nn.Module
        The network to validate.
    loss_criterion : torch.nn.Module
        The loss criterion to use.
    val_dataloader : torch.utils.data.DataLoader
        The validation data loader.
    epoch : int
        The epoch number.

    Returns
    -------
    avg_loss : float
        The average loss for the epoch.
    avg_ssim : float
        The average SSIM for the epoch.
    """
    avg_loss = 0.0
    avg_ssim = 0.0
    ssim_criterion = SSIM()
    device = get_device()
    radial_normalizer = batch_norm(
        in_channels =config.in_channels,
    )
    mag = lambda x: (x[..., 0] ** 2 + x[..., 1] ** 2) ** 0.5
    with torch.no_grad():
        for itt, (input, target) in enumerate(val_dataloader):
            X = Variable(torch.FloatTensor(input.float())).to(device)
            y = Variable(torch.FloatTensor(target.float())).to(device)

            if config.normalize_input:
                X = radial_normalizer(X)
                y = radial_normalizer(y)

            y_pred = net(X)

            loss = loss_criterion(y_pred, y)
            ssim = ssim_criterion(mag(y_pred), mag(y))

            avg_loss += loss.detach().item() / len(val_dataloader)
            avg_ssim += ssim.detach().item() / len(val_dataloader)
            
            logging.info('Epoch: {0} - Itter: {1}/{2} - loss: {3:.6f} - SSIM: {4:.6f}'.format(
                epoch, itt, len(val_dataloader), loss.detach().item(), ssim.detach().item())
            )
    return avg_loss, avg_ssim


def train(net, optimizer, loss_criterion, tr_dataloader, val_dataloader):
    """Train the network

    Parameters
    ----------
    net : torch.nn.Module
        The network to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    loss_criterion : torch.nn.Module
        The loss criterion to use.
    tr_dataloader : torch.utils.data.DataLoader
        The training data loader.
    val_dataloader : torch.utils.data.DataLoader
        The validation data loader.
    """
    best_loss = inf
    for epoch in range(config.num_epochs):
        logging.info(f'Training epoch {epoch}/{config.num_epochs}...')

        optimizer = adjust_learning_rate(epoch, optimizer)

        # Training
        avg_tr_loss, net, optimizer = train_epoch(
            net, optimizer, loss_criterion, tr_dataloader, epoch
        )
        logging.info(f'Epoch {epoch} - Avg. training loss: {avg_tr_loss:.3f}')

        # Validation
        avg_vld_loss, avg_vld_ssim = validate(net, loss_criterion, val_dataloader, epoch)
        logging.info(f'Epoch {epoch} - Avg. validation loss: {avg_tr_loss:.3f}, SSIM: {avg_vld_ssim:.3f}')

        save_checkpoint(
            {
                'epoch': epoch,
                'arch': 'complexnet',
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            is_best = avg_vld_loss < best_loss,
            filename = config.models_dir + 'checkpoint.pth'
        )
        logging.info('Model Saved!')


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Save a checkpoint

    Parameters
    ----------
    state : dict
        The state to save.
    is_best : bool
        Whether this is the best model.
    filename : str
        The filename to save the checkpoint to.
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


def adjust_learning_rate(epoch, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs
    
    Parameters
    ----------
    epoch : int
        The epoch number.
    optimizer : torch.optim.Optimizer
        The optimizer.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        The optimizer.
    """
    lr = config.learning_rate * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


if __name__ == '__main__':
    set_seeds(222)
    tr_dataloader, val_dataloader = get_dataloaders()
    net = CUNet(config.in_channels, config.out_channels)
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()

    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
    loss_criterion = Loss.MSELoss()
    os.makedirs(config.models_dir, exist_ok=True)

    train(net, optimizer, loss_criterion, tr_dataloader, val_dataloader)
