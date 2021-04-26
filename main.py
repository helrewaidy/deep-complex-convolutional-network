
import shutil
import sys

import torch.nn.modules.loss as Loss
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from torch import optim

from parameters import Parameters
from unet import UNet
from utils.cmplx_batchnorm import magnitude, normalize_complex_batch_by_magnitude_only
from utils.dataset import *
from utils.myloss import *

# set seed points
seed_num = 888
torch.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
np.random.seed(seed_num)
params = Parameters()

####################################
# Create Data Generators
training_DG, validation_DG, params = get_dataset_generators(params)

# Create Model
net = UNet(params.n_channels, 1)

if params.multi_GPU:
    net = torch.nn.DataParallel(net, device_ids=params.device_ids).cuda()
else:
    net.to(params.device)

optimizer = optim.Adam(net.parameters(), lr=params.args.lr)

if not os.path.exists(params.model_save_dir):
    os.makedirs(params.model_save_dir)

if not os.path.exists(params.tensorboard_dir):
    os.makedirs(params.tensorboard_dir)

writer = SummaryWriter(params.tensorboard_dir)


def train():
    # INITIALIZATIONS
    tr_loss = list()
    tr_ssim_loss = list()
    ssimCriterion = SSIM()
    mseCriterion = Loss.MSELoss()

    lossCrit = mseCriterion
    vld_mse_loss = list()
    vld_ssim_loss = list()

    vld_mse_loss_in = list()
    vld_ssim_loss_in = list()
    vi = 0
    i = 0

    # LOAD LATEST (or SPECIFIC) MODEL
    s_epoch = load_model(-1)

    for epoch in range(s_epoch, params.epochs):
        print('epoch {}/{}...'.format(epoch + 1, params.epochs))

        adjust_learning_rate(epoch)

        ###########################################
        # Training
        l = 0
        itt = 0
        TAG = 'Training'
        if not params.Validation_Only:
            for local_batch, local_labels, sliceID, orig_size, usr in training_DG:

                X = Variable(torch.FloatTensor(local_batch.float())).to(params.device)
                y = Variable(torch.FloatTensor(local_labels.float())).to(params.device)

                input_mag = normalize_batch_torch(get_magnitude(X))
                LOST_mag = normalize_batch_torch(y[:, :, :, :, 0])

                if params.complex_net:
                    X = normalize_complex_batch_by_magnitude_only(X, False)
                    y = normalize_complex_batch_by_magnitude_only(y, True)
                else:
                    X = get_magnitude(X)
                    y = y[:, :, :, :, 0]
                    X = normalize_batch_torch(X)
                    y = normalize_batch_torch(y)

                y_pred = net(X)

                if params.complex_net:
                    loss = lossCrit(magnitude(y_pred).squeeze(1), y[:, :, :, :, 0].squeeze(1))
                    simloss = ssimCriterion(magnitude(y_pred), y[:, :, :, :, 0])
                else:
                    loss = lossCrit(y_pred, y)
                    simloss = ssimCriterion(y_pred, y)

                tr_loss.append(loss.cpu().data.numpy())
                tr_ssim_loss.append(simloss.cpu().data.numpy())

                l += loss.data[0]

                optimizer.zero_grad()
                loss.backward()
                i += 1
                optimizer.step()

                inloss = mseCriterion(input_mag, LOST_mag)

                print('Epoch: {0} - {1:.3f}%'.format(epoch + 1, 100 * (itt * params.batch_size) / len(
                    training_DG.dataset.input_IDs))
                      + ' \tIter: ' + str(i)
                      + '\tLoss: {0:.6f}'.format(loss.data[0])
                      + '\tInputLoss: {0:.6f}'.format(inloss.data[0]))
                itt += 1

                if itt % 100 == 0:
                    is_best = 0
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'loss': tr_loss,
                        'arch': 'recoNet_Model1',
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, is_best, filename=params.model_save_dir + 'MODEL_EPOCH{}.pth'.format(epoch + 1))

                    print('Model Saved!')
            avg_loss = params.batch_size * l / len(training_DG.dataset.input_IDs)
            print('Total Loss : {0:.6f} \t Avg. Loss {1:.6f}'.format(l, avg_loss))

        else:
            load_model(epoch + 1)

        #####################################
        # Validation

        vitt = 0
        vld_mse = 0
        vld_ssim = 0
        vld_psnr = 0
        vld_mse_in = 0
        vld_ssim_in = 0
        vld_psnr_in = 0

        TAG = 'Validation'
        with torch.no_grad():
            for local_batch, local_labels, sliceID, orig_size, usr in validation_DG:
                X = Variable(torch.FloatTensor(local_batch.float())).to(params.device)
                y = Variable(torch.FloatTensor(local_labels.float())).to(params.device)

                input_mag = normalize_batch_torch(get_magnitude(X))
                LOST_mag = normalize_batch_torch(y[:, :, :, :, 0])

                if params.complex_net:
                    X = normalize_complex_batch_by_magnitude_only(X, False)
                    y = normalize_complex_batch_by_magnitude_only(y, True)
                else:
                    X = get_magnitude(X)
                    y = y[:, :, :, :, 0]
                    X = normalize_batch_torch(X)
                    y = normalize_batch_torch(y)

                y_pred = net(X)

                if params.complex_net:
                    mseloss = mseCriterion(magnitude(y_pred).squeeze(1), y[:, :, :, :, 0].squeeze(1))
                    ssimloss = ssimCriterion(magnitude(y_pred), y[:, :, :, :, 0])

                else:
                    mseloss = mseCriterion(y_pred, y)
                    ssimloss = ssimCriterion(y_pred, y)

                mseloss_in = mseCriterion(input_mag, LOST_mag)
                ssimloss_in = ssimCriterion(input_mag, LOST_mag)

                vld_mse_loss.append(mseloss.cpu().data.numpy())
                vld_ssim_loss.append(ssimloss.cpu().data.numpy())

                vld_mse_loss_in.append(mseloss_in.cpu().data.numpy())
                vld_ssim_loss_in.append(ssimloss_in.cpu().data.numpy())

                vld_mse += mseloss.data[0]
                vld_ssim += ssimloss.data[0]

                vld_mse_in += mseloss_in.data[0]
                vld_ssim_in += ssimloss_in.data[0]

                vi += 1
                vitt += 1

                if params.complex_net:
                    inloss = mseCriterion(magnitude(X).squeeze(1), y[:, :, :, :, 0].squeeze(1))
                else:
                    inloss = mseCriterion(X, y)

                print('Epoch: {0} - {1:.3f}%'.format(epoch + 1, 100 * (vitt * params.batch_size) / len(
                    validation_DG.dataset.input_IDs))
                      + ' \tIter: ' + str(vi)
                      + '\tSME: {0:.6f}'.format(mseloss.data[0])
                      + '\tSSIM: {0:.6f}'.format(ssimloss.data[0])
                      + '\tInputLoss: {0:.6f}'.format(inloss.data[0]))

            avg_factor = params.batch_size / len(validation_DG.dataset.input_IDs)
            print('Avg. MSE : {0:.6f}'.format(vld_mse * avg_factor)
                  + '\tAvg. SSIM : {0:.6f}'.format(vld_ssim * avg_factor)
                  + '\tAvg. PSNR : {0:.6f}'.format(vld_psnr * avg_factor)
                  + 'Avg. Input_MSE : {0:.6f}'.format(vld_mse_in * avg_factor)
                  + '\tAvg. Input_SSIM : {0:.6f}'.format(vld_ssim_in * avg_factor)
                  + '\tAvg. Input_PSNR : {0:.6f}'.format(vld_psnr_in * avg_factor)
                  )

    writer.close()


def load_model(epoch=0):
    '''
    load model at specific epoch
    :param:
    epoch is integer specify which model to be loaded. If
        epoch = -1: load latest model or start from 1 if there is no saved models
        epoch = 0: don't load any model and start from model #1
        epoch = num: load model #num

    :returns: the loeaded model
    '''
    models = [m for m in os.listdir(params.model_save_dir) if m.endswith('.pth')]
    if len(models) == 0:
        return 1

    s_epoch = epoch if epoch != -1 else max([int(epo.split('_')[-1][:-4]) for epo in models[:]])
    try:
        model = torch.load(params.model_save_dir + models[0][0:11] + str(s_epoch) + '.pth')
        model.load_state_dict(model['state_dict'])
        optimizer.load_state_dict(model['optimizer'])
    except:
        print('Model {0} does not exist!'.format(s_epoch))
        return 1
    return s_epoch


def get_magnitude(input):
    return (input[:, :, :, :, 0] ** 2 + input[:, :, :, :, 1] ** 2) ** 0.5


def normalize_batch_torch(p, dims=[2, 3]):
    ''' normalize each slice alone'''
    if torch.std(p) == 0:
        raise ZeroDivisionError
    shape = p.shape
    if p.ndimension() == 4:
        pv = p.reshape([shape[0], shape[1], shape[2] * shape[3]])
    else:
        raise NotImplementedError

    mean = pv.mean(dim=2, keepdim=True).unsqueeze(p.ndimension() - 1)
    std = pv.std(dim=2, keepdim=True).unsqueeze(p.ndimension() - 1)

    return (p - mean) / std


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = params.args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    params.parse_args()
    try:
        train(net)
    except KeyboardInterrupt:
        print('Interrupted')
        torch.save(net.state_dict(), 'MODEL_INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
