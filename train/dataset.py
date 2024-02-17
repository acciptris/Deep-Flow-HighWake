import os
import random

import numpy as np
from torch.utils.data import Dataset

# global switch, use fixed max values for dim-less airfoil data?
fixedAirfoilNormalization = False
# global switch, make data dimensionless?
makeDimLess = True
# global switch, remove constant offsets from pressure channel?
removePOffset = True

## helper - compute absolute of inputs or targets
def find_absmax(data, use_targets, x):
    maxval = 0
    for i in range(data.totalLength):
        if use_targets == 0:
            temp_tensor = data.inputs[i]
        else:
            temp_tensor = data.targets[i]
        temp_max = np.max(np.abs(temp_tensor[x]))
        if temp_max > maxval:
            maxval = temp_max
    return maxval


######################################## DATA LOADER #########################################
#         also normalizes data with max , and optionally makes it dimensionless              #


def LoaderNormalizer(data, dataProp, isTest=False, shuffle=0):
    """
    # data: pass TurbDataset object with initialized dataDir / dataDirTest paths
    # train: when off, process as test data (first load regular for normalization if needed, then replace by test data)
    # dataProp: proportions for loading & mixing 3 different data directories "reg", "shear", "sup"
    #           should be array with [total-length, fraction-regular, fraction-superimposed, fraction-sheared],
    #           passing None means off, then loads from single directory
    """

    directory_name = data.dataDir
    directory_name_split = directory_name.split(sep="/")
    maxSaveFile = directory_name_split[-2] + "_max_values.txt"

    if os.path.exists(maxSaveFile) and isTest:
        print(f"Loading max values from {maxSaveFile}")
        with open(maxSaveFile, "r") as file:
            lines = file.readlines()

        data.max_inputs_0 = float(lines[0][:-1])
        data.max_inputs_1 = float(lines[1][:-1])
        data.max_inputs_2 = float(lines[2][:-1])
        data.max_targets_0 = float(lines[3][:-1])
        data.max_targets_1 = float(lines[4][:-1])
        data.max_targets_2 = float(lines[5][:-1])

    else:
        if len(dataProp) == 1:
            # load single directory
            files = os.listdir(data.dataDir)
            files.sort()
            for i in range(shuffle):
                random.shuffle(files)
            # if isTest:
            #     print("Reducing data to load for tests")
            #     files = files[0:min(10, len(files))]
            if dataProp[0] is None:
                data.totalLength = len(files)
            else:
                data.totalLength = dataProp[0]
            data.inputs = np.empty((data.totalLength, 3, 128, 128))
            data.targets = np.empty((data.totalLength, 3, 128, 128))

            for i in range(data.totalLength):
                npfile = np.load(data.dataDir + files[i])
                d = npfile["a"]
                data.inputs[i] = d[0:3]
                data.targets[i] = d[3:6]
            print("Number of data loaded:", len(data.inputs))

        else:

            # load from folders reg, sup, and shear under the folder dataDir

            files1 = os.listdir(data.dataDir + "reg/")
            files1.sort()
            files2 = os.listdir(data.dataDir + "sup/")
            files2.sort()
            files3 = os.listdir(data.dataDir + "shear/")
            files3.sort()
            for i in range(shuffle):
                random.shuffle(files1)
                random.shuffle(files2)
                random.shuffle(files3)

            if isTest:
                data.totalLength = len(files1) + len(files2) + len(files3)
            else:
                data.totalLength = int(dataProp[0])

            data.inputs = np.empty((data.totalLength, 3, 128, 128))
            data.targets = np.empty((data.totalLength, 3, 128, 128))

            temp_1, temp_2 = 0, 0

            if isTest:
                for i, file in enumerate(files1, 0):
                    npfile = np.load(data.dataDir + "reg/" + file)
                    d = npfile["a"]
                    data.inputs[i] = d[0:3]
                    data.targets[i] = d[3:6]
                for i, file in enumerate(files2, len(files1)):
                    npfile = np.load(data.dataDir + "sup/" + file)
                    d = npfile["a"]
                    data.inputs[i] = d[0:3]
                    data.targets[i] = d[3:6]
                for i, file in enumerate(files3, len(files1) + len(files2)):
                    npfile = np.load(data.dataDir + "shear/" + file)
                    d = npfile["a"]
                    data.inputs[i] = d[0:3]
                    data.targets[i] = d[3:6]
                print(
                    "Number of data loaded (reg, sup, shear):",
                    len(files1),
                    len(files2),
                    len(files3),
                )

            else:
                for i in range(data.totalLength):
                    if i >= (1 - dataProp[3]) * dataProp[0]:
                        npfile = np.load(data.dataDir + "shear/" + files3[i - temp_2])
                        d = npfile["a"]
                        data.inputs[i] = d[0:3]
                        data.targets[i] = d[3:6]
                    elif i >= (dataProp[1]) * dataProp[0]:
                        npfile = np.load(data.dataDir + "sup/" + files2[i - temp_1])
                        d = npfile["a"]
                        data.inputs[i] = d[0:3]
                        data.targets[i] = d[3:6]
                        temp_2 = i + 1
                    else:
                        npfile = np.load(data.dataDir + "reg/" + files1[i])
                        d = npfile["a"]
                        data.inputs[i] = d[0:3]
                        data.targets[i] = d[3:6]
                        temp_1 = i + 1
                        temp_2 = i + 1
                print(
                    "Number of data loaded (reg, sup, shear):",
                    temp_1,
                    temp_2 - temp_1,
                    i + 1 - temp_2,
                )

        ################################## NORMALIZATION OF TRAINING DATA ##########################################

        if removePOffset:
            for i in range(data.totalLength):
                data.targets[i, 0, :, :] -= np.mean(
                    data.targets[i, 0, :, :]
                )  # remove offset
                data.targets[i, 0, :, :] -= (
                    data.targets[i, 0, :, :] * data.inputs[i, 2, :, :]
                )  # pressure * mask

        # make dimensionless based on current data set
        if makeDimLess:
            for i in range(data.totalLength):
                # only scale outputs, inputs are scaled by max only
                v_norm = (
                    np.max(np.abs(data.inputs[i, 0, :, :])) ** 2
                    + np.max(np.abs(data.inputs[i, 1, :, :])) ** 2
                ) ** 0.5
                data.targets[i, 0, :, :] /= v_norm ** 2
                data.targets[i, 1, :, :] /= v_norm
                data.targets[i, 2, :, :] /= v_norm

        # normalize to -1..1 range, from min/max of predefined
        if fixedAirfoilNormalization:
            # hard coded maxima , inputs dont change
            data.max_inputs_0 = 100.0
            data.max_inputs_1 = 38.12
            data.max_inputs_2 = 1.0

            # targets depend on normalization
            if makeDimLess:
                data.max_targets_0 = 4.65
                data.max_targets_1 = 2.04
                data.max_targets_2 = 2.37
                print(
                    "Using fixed maxima "
                    + format(
                        [data.max_targets_0, data.max_targets_1, data.max_targets_2]
                    )
                )
            else:  # full range
                data.max_targets_0 = 40000.0
                data.max_targets_1 = 200.0
                data.max_targets_2 = 216.0
                print(
                    "Using fixed maxima "
                    + format(
                        [data.max_targets_0, data.max_targets_1, data.max_targets_2]
                    )
                )

        else:  # use current max values from loaded data
            print("Finding max values")
            data.max_inputs_0 = find_absmax(data, 0, 0)
            data.max_inputs_1 = find_absmax(data, 0, 1)
            data.max_inputs_2 = find_absmax(data, 0, 2)  # mask, not really necessary
            data.max_targets_0 = find_absmax(data, 1, 0)
            data.max_targets_1 = find_absmax(data, 1, 1)
            data.max_targets_2 = find_absmax(data, 1, 2)

            if isTest:
                print("Saving max values")
                with open(maxSaveFile, "w") as file:
                    print(data.max_inputs_0, file=file)
                    print(data.max_inputs_1, file=file)
                    print(data.max_inputs_2, file=file)
                    print(data.max_targets_0, file=file)
                    print(data.max_targets_1, file=file)
                    print(data.max_targets_2, file=file)

        data.inputs[:, 0, :, :] *= 1.0 / data.max_inputs_0
        data.inputs[:, 1, :, :] *= 1.0 / data.max_inputs_1

        data.targets[:, 0, :, :] *= 1.0 / data.max_targets_0
        data.targets[:, 1, :, :] *= 1.0 / data.max_targets_1
        data.targets[:, 2, :, :] *= 1.0 / data.max_targets_2

    print(
        "Maxima inputs "
        + format([data.max_inputs_0, data.max_inputs_1, data.max_inputs_2])
    )
    print(
        "Maxima targets "
        + format([data.max_targets_0, data.max_targets_1, data.max_targets_2])
    )

    ###################################### NORMALIZATION  OF TEST DATA #############################################

    if isTest:
        files = os.listdir(data.dataDirTest)
        files.sort()
        data.totalLength = len(files)
        data.inputs = np.empty((len(files), 3, 128, 128))
        data.targets = np.empty((len(files), 3, 128, 128))
        for i, file in enumerate(files):
            npfile = np.load(data.dataDirTest + file)
            d = npfile["a"]
            data.inputs[i] = d[0:3]
            data.targets[i] = d[3:6]

        if removePOffset:
            for i in range(data.totalLength):
                data.targets[i, 0, :, :] -= np.mean(
                    data.targets[i, 0, :, :]
                )  # remove offset
                data.targets[i, 0, :, :] -= (
                    data.targets[i, 0, :, :] * data.inputs[i, 2, :, :]
                )  # pressure * mask

        if makeDimLess:
            for i in range(len(files)):
                v_norm = (
                    np.max(np.abs(data.inputs[i, 0, :, :])) ** 2
                    + np.max(np.abs(data.inputs[i, 1, :, :])) ** 2
                ) ** 0.5
                data.targets[i, 0, :, :] /= v_norm ** 2
                data.targets[i, 1, :, :] /= v_norm
                data.targets[i, 2, :, :] /= v_norm

        data.inputs[:, 0, :, :] *= 1.0 / data.max_inputs_0
        data.inputs[:, 1, :, :] *= 1.0 / data.max_inputs_1

        data.targets[:, 0, :, :] *= 1.0 / data.max_targets_0
        data.targets[:, 1, :, :] *= 1.0 / data.max_targets_1
        data.targets[:, 2, :, :] *= 1.0 / data.max_targets_2

    print(
        "Data stats, input  mean %f, max  %f;   targets mean %f , max %f "
        % (
            np.mean(np.abs(data.targets), keepdims=False),
            np.max(np.abs(data.targets), keepdims=False),
            np.mean(np.abs(data.inputs), keepdims=False),
            np.max(np.abs(data.inputs), keepdims=False),
        )
    )

    return data


######################################## DATA SET CLASS #########################################


class TurbDataset(Dataset):

    # mode "enum" , pass to mode param of TurbDataset (note, validation mode is not necessary anymore)
    TRAIN = 0
    TEST = 2

    def __init__(
        self,
        dataProp,
        mode=TRAIN,
        dataDir="../data/train/",
        dataDirTest="../data/test/",
        shuffle=0,
        normMode=0,
    ):
        global makeDimLess, removePOffset
        """
        :param dataProp: for split&mix from multiple dirs, see LoaderNormalizer; None means off
        :param mode: TRAIN|TEST , toggle regular 80/20 split for training & validation data, or load test data
        :param dataDir: directory containing training data
        :param dataDirTest: second directory containing test data , needs training dir for normalization
        :param normMode: toggle normalization
        """
        if not (mode == self.TRAIN or mode == self.TEST):
            print("Error - TurbDataset invalid mode " + format(mode))
            exit(1)

        if normMode == 1:
            print("Warning - poff off!!")
            removePOffset = False
        if normMode == 2:
            print("Warning - poff and dimless off!!!")
            makeDimLess = False
            removePOffset = False

        self.mode = mode
        self.dataDir = dataDir
        self.dataDirTest = dataDirTest  # only for mode==self.TEST

        # load & normalize data
        self = LoaderNormalizer(
            self, dataProp, isTest=(mode == self.TEST), shuffle=shuffle
        )

        if not self.mode == self.TEST:
            # split for train/validation sets (80/20) , max 400
            targetLength = self.totalLength - min(int(self.totalLength * 0.2), 400)

            self.valiInputs = self.inputs[targetLength:]
            self.valiTargets = self.targets[targetLength:]
            self.valiLength = self.totalLength - targetLength

            self.inputs = self.inputs[:targetLength]
            self.targets = self.targets[:targetLength]
            self.totalLength = self.inputs.shape[0]

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    #  reverts normalization
    def denormalize(self, data, v_norm):
        a = data.copy()
        a[0, :, :] /= 1.0 / self.max_targets_0
        a[1, :, :] /= 1.0 / self.max_targets_1
        a[2, :, :] /= 1.0 / self.max_targets_2

        if makeDimLess:
            a[0, :, :] *= v_norm ** 2
            a[1, :, :] *= v_norm
            a[2, :, :] *= v_norm
        return a


# simplified validation data set (main one is TurbDataset above)


class ValiDataset(TurbDataset):
    def __init__(self, dataset):
        self.inputs = dataset.valiInputs
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

