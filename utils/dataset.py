from torch.utils import data
from scipy.io import loadmat
import numpy as np
from utils.saveNet import *


# params = Parameters()

def resizeImage(img, newSize, Interpolation=False):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)

    if Interpolation:
        return imresize(img, tuple(newSize), interp='bilinear')
    else:

        x1 = (img.shape[0] - newSize[0]) // 2
        x2 = img.shape[0] - newSize[0] - x1

        y1 = (img.shape[1] - newSize[1]) // 2
        y2 = img.shape[1] - newSize[1] - y1

        if img.ndim == 3:
            if x1 > 0:
                img = img[x1:-x2, :, :]
            elif x1 < 0:
                img = np.pad(img, ((-x1, -x2), (0, 0), (0, 0)), 'constant')  # ((top, bottom), (left, right))

            if y1 > 0:
                img = img[:, y1:-y2, :]
            elif y1 < 0:
                img = np.pad(img, ((0, 0), (-y1, -y2), (0, 0)), 'constant')  # ((top, bottom), (left, right))

        elif img.ndim == 4:
            if x1 > 0:
                img = img[x1:-x2, :, :, :]
            elif x1 < 0:
                img = np.pad(img, ((-x1, -x2), (0, 0), (0, 0), (0, 0)), 'constant')  # ((top, bottom), (left, right))

            if y1 > 0:
                img = img[:, y1:-y2, :, :]
            elif y1 < 0:
                img = np.pad(img, ((0, 0), (-y1, -y2), (0, 0), (0, 0)), 'constant')  # ((top, bottom), (left, right))
        return img.squeeze()


def getDatasetGenerators(params):
    def getPatientSlicesURLs(patient_url):
        islices = list()
        oslices = list()
        for fs in os.listdir(patient_url + '/InputData/Input_realAndImag/'):
            islices.append(patient_url + '/InputData/Input_realAndImag/' + fs)

        for fs in os.listdir(patient_url + '/CSRecon/CSRecon_Data_small/'):
            oslices.append(patient_url + '/CSRecon/CSRecon_Data_small/' + fs)
        islices = sorted(islices, key=lambda x: int((x.rsplit(sep='/')[-1])[8:-4]))
        oslices = sorted(oslices, key=lambda x: int((x.rsplit(sep='/')[-1])[8:-4]))

        return (islices, oslices)

    num_slices_per_patient = []
    params.input_slices = []
    params.groundTruth_slices = []
    params.us_rates = [];

    P = loadmat(params.net_save_dir + 'lgePatients_urls.mat')['lgePatients']
    pNames = [i[0][0] for i in P]
    usRates = [i[1][0] for i in P]

    k = -1
    for p in pNames:
        k += 1
        for dir in params.dir:
            pdir = dir + p
            if os.path.exists(pdir):
                params.patients.append(pdir)
                slices = getPatientSlicesURLs(pdir)
                num_slices_per_patient.append(len(slices[0]))
                params.input_slices = np.concatenate((params.input_slices, slices[0]))
                params.groundTruth_slices = np.concatenate((params.groundTruth_slices, slices[1]))
                params.us_rates = np.concatenate([params.us_rates, usRates[k] * np.ones(len(slices[0]))])
                continue

    print('-- Number of Datasets: ' + str(len(params.patients)))

    params.num_slices_per_patient = num_slices_per_patient

    training_ptns = round(params.training_percent * len(num_slices_per_patient))

    training_end_indx = sum(num_slices_per_patient[0:training_ptns])
    evaluation_end_indx = training_end_indx + sum(num_slices_per_patient)

    params.training_patients_index = range(0, training_ptns)

    training_DS = DataGenerator(input_IDs=params.input_slices[:training_end_indx],
                                output_IDs=params.groundTruth_slices[:training_end_indx],
                                undersampling_rates=params.us_rates[:training_end_indx],
                                dim=(params.img_size[0], params.img_size[1], 2),
                                n_channels=params.n_channels)

    validation_DS = DataGenerator(input_IDs=params.input_slices[training_end_indx:evaluation_end_indx],
                                  output_IDs=params.groundTruth_slices[training_end_indx:evaluation_end_indx],
                                  undersampling_rates=params.us_rates[training_end_indx:evaluation_end_indx],
                                  dim=(params.img_size[0], params.img_size[1], 2),
                                  n_channels=params.n_channels)

    training_DL = data.DataLoader(training_DS, batch_size=params.batch_size, shuffle=True,
                                  num_workers=params.data_loders_num_workers)

    validation_DL = data.DataLoader(validation_DS, batch_size=params.batch_size, shuffle=False,
                                    num_workers=params.data_loders_num_workers)

    return training_DL, validation_DL, params


class DataGenerator(data.Dataset):
    'Generates data for Keras'

    def __init__(self, input_IDs, output_IDs, undersampling_rates=None, dim=(256, 256, 2), n_channels=1):
        'Initialization'
        self.dim = dim
        self.output_IDs = output_IDs
        self.input_IDs = input_IDs
        self.n_channels = n_channels
        self.undersampling_rates = undersampling_rates

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.input_IDs)

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        X, y, orig_size = self.__data_generation(self.input_IDs[index], self.output_IDs[index])
        if self.undersampling_rates is not None:
            usr = self.undersampling_rates[index]
        else:
            usr = None

        return X, y, self.input_IDs[index], orig_size, usr

    def __data_generation(self, input_IDs_temp, output_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.n_channels, *self.dim))
        y = np.zeros((self.n_channels, *self.dim))

        # Generate data
        img = loadmat(input_IDs_temp)['Input_realAndImag']
        orig_size = [img.shape[0], img.shape[1]]
        X[0, ] = resizeImage(img, [self.dim[0], self.dim[1]])
        y[0, :, :, 0] = resizeImage(loadmat(output_IDs_temp)['Data'], [self.dim[0], self.dim[1]])
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        return X, y, orig_size


