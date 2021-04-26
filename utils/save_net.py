import os
from parameters import Parameters
from scipy.io.matlab.mio import savemat

params = Parameters()


def saveTensorToMat(x, varName='x', fileName='', save_dir=params.net_save_dir):
    '''
    x: is the variable to be saved
    name: is the name of the file and name of variable in the mat file
    '''
    if fileName is '':
        fileName = varName

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    savemat(save_dir + fileName + '.mat', {varName: x.cpu().data.numpy()})


def saveArrayToMat(x, varName='x', fileName='', save_dir=params.net_save_dir):
    '''
    x: is the variable to be saved
    name: is the name of the file and name of variable in the mat file
    '''
    if fileName is '':
        fileName = varName

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    savemat(save_dir + fileName + '.mat', {varName: x})
