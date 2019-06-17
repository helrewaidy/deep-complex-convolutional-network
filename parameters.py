'''
Created on May 17, 2018

@author: helrewaidy
'''
# models

import argparse

import torch

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

########################## Initializations ########################################
model_names = 'recoNet_Model1'
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--cpu', '-c', action='store_true',
                    help='Do not use the cuda version of the net',
                    default=False)
parser.add_argument('--viz', '-v', action='store_true',
                    help='Visualize the images as they are processed',
                    default=False)
parser.add_argument('--no-save', '-n', action='store_false',
                    help='Do not save the output masks',
                    default=False)
parser.add_argument('--model', '-m', default='MODEL_EPOCH417.pth',
                    metavar='FILE',
                    help='Specify the file in which is stored the model'
                         " (default : 'MODEL.pth')")


###################################################################
class Parameters():
    def __init__(self):
        super(Parameters, self).__init__()

        ## Hardware/GPU parameters =================================================
        self.Op_Node = 'spider'  # 'alpha_V12' # 'myPC', 'O2', 'spider'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tbVisualize = False
        self.tbVisualize_kernels = False
        self.tbVisualize_featuremaps = False
        self.multi_GPU = True

        if self.Op_Node in ['myPC', 'alpha_V12']:
            self.device_ids = [0]
        elif self.Op_Node in ['spider', 'O2']:
            self.device_ids = range(0, torch.cuda.device_count())

        if self.Op_Node in ['spider', 'O2', 'alpha_V12']:
            self.data_loders_num_workers = 20
        else:
            self.data_loders_num_workers = 4

        ## Network/Model parameters =================================================
        if self.Op_Node in ['myPC', 'alpha_V12']:
            self.batch_size = 2
        elif self.Op_Node in ['spider', 'O2']:
            self.batch_size = 100 * len(self.device_ids) // 8

        print('-- # GPUs: ', len(self.device_ids))
        print('-- batch_size: ', self.batch_size)
        self.args = parser.parse_args()

        self.activation_func = 'CReLU'  # 'CReLU' # 'modReLU' 'KAF2D' 'ZReLU'
        self.args.lr = 0.1
        self.dropout_ratio = 0.0
        self.epochs = 10
        self.training_percent = 0.7
        self.nIterations = 1
        self.magnitude_only = False
        self.Validation_Only = False
        self.Evaluation = False

        #########
        # self.MODEL = 0 # Original U-net implementation
        self.MODEL = 3  # Complex U-net (URUS)
        #         self.MODEL = 3.1 # Complex stacked convolution layers
        #         self.MODEL = 3.2 # Complex U-net with different kernel configuration
        #        self.MODEL = 4 # Complex U-Net with residual connection
        #         self.MODEL = 7 # Real shallow U-net layer [double size] (magNet)

        #########
        if self.MODEL in [2, 3, 3.1, 3.2, 4, 5, 6]:
            self.complex_net = True
        else:
            self.complex_net = False

        ## Dataset and paths =================================================

        self.ds_total_num_slices = 0
        self.patients = []
        self.Rate = 3
        self.input_slices = list()
        self.num_slices_per_patient = list()
        self.groundTruth_slices = list()
        self.training_patients_index = list()
        self.us_rates = list()
        self.saveVolumeData = False
        self.multiCoilInput = False
        self.coilCombinedInputTV = True
        self.img_size = [256, 256]
        self.n_channels = 1

        self.cropped_dataset64 = False
        if self.cropped_dataset64:
            crop_txt = '_cropped64'
        else:
            crop_txt = ''
        self.trialNum = '3.555'
        self.arch_name = 'Model_0' + str(
            self.MODEL) + '_R' + str(
            self.Rate) + 'Trial' + self.trialNum

        self.dir = {'./ReconData_coilCombTVDL/Rate_' + str(
            self.Rate) + crop_txt + '/',
                    './ReconData_coilCombTVDL/Rate_' + str(
                        self.Rate) + '/'
                    }
        self.model_save_dir = './RecoNet-Model/' + self.arch_name + '/'
        self.net_save_dir = './MatData/'
        self.tensorboard_dir = './RecoNet-Model/' + self.arch_name + '_tensorboard/'

        self.args.model = self.model_save_dir + 'MODEL_EPOCH.pth'















