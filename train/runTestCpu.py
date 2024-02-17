import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils
from dataset import TurbDataset
from DfpNet import TurbNetG, weights_init
from utils import log

start_time = time.time()

#####################################

suffix = ""  # customize loading & output if necessary
prefix = ""

train_directory = "../data/train/"
test_directory = "../data/test/"

if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

expo = 3

save_outputs = True

####################################

print("Loading Data...")
dataset = TurbDataset(
    [None], mode=TurbDataset.TEST, dataDirTest=test_directory, dataDir=train_directory
)
testLoader = DataLoader(dataset, batch_size=1, shuffle=False)

targets = torch.FloatTensor(1, 3, 128, 128)
targets = Variable(targets)
inputs = torch.FloatTensor(1, 3, 128, 128)
inputs = Variable(inputs)

targets_dn = torch.FloatTensor(1, 3, 128, 128)
targets_dn = Variable(targets_dn)
outputs_dn = torch.FloatTensor(1, 3, 128, 128)
outputs_dn = Variable(outputs_dn)

netG = TurbNetG(channelExponent=expo)

test_directory_name_split = test_directory.split(sep="/")
lf = (
    "./"
    + prefix
    + "_"
    + test_directory_name_split[-2]
    + "_testout{}.txt".format(suffix)
)
# utils.makeDirs(["results_test"])

# loop over different trained models
avgLoss = 0.0
losses = []
models = []

test_files = os.listdir(test_directory)
test_files.sort()

for si in range(25):
    s = chr(96 + si)
    if si == 0:
        s = ""  # check modelG, and modelG + char
    modelFn = "./" + prefix + "_modelG_{}{}".format(suffix, s)
    if not os.path.isfile(modelFn):
        continue

    models.append(modelFn)
    log(lf, "Loading " + modelFn)
    netG.load_state_dict(torch.load(modelFn))
    log(lf, "Loaded " + modelFn)

    log(lf, "Test directory: " + test_directory)

    criterionL1 = nn.L1Loss()
    L1val_accum = 0.0
    L1val_dn_accum = 0.0
    lossPer_p_accum = 0
    lossPer_v_accum = 0
    lossPer_accum = 0

    netG.eval()

    utils.makeDirs([prefix + "_results_test_" + suffix + s])
    for i, data in enumerate(testLoader, 0):
        inputs_cpu, targets_cpu = data
        targets_cpu, inputs_cpu = targets_cpu.float(), inputs_cpu.float()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()[0]
        targets_cpu = targets_cpu.cpu().numpy()[0]

        lossL1 = criterionL1(outputs, targets)
        L1val_accum += lossL1.item()

        # precentage loss by ratio of means which is same as the ratio of the sum
        lossPer_p = np.sum(np.abs(outputs_cpu[0] - targets_cpu[0])) / np.sum(
            np.abs(targets_cpu[0])
        )
        lossPer_v = (
            np.sum(np.abs(outputs_cpu[1] - targets_cpu[1]))
            + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2]))
        ) / (np.sum(np.abs(targets_cpu[1])) + np.sum(np.abs(targets_cpu[2])))
        lossPer = np.sum(np.abs(outputs_cpu - targets_cpu)) / np.sum(
            np.abs(targets_cpu)
        )
        lossPer_p_accum += lossPer_p.item()
        lossPer_v_accum += lossPer_v.item()
        lossPer_accum += lossPer.item()

        log(lf, f"Test sample {i} : {test_files[i]}")
        log(
            lf,
            "    pressure:  abs. difference, ratio: %f , %f "
            % (np.sum(np.abs(outputs_cpu[0] - targets_cpu[0])), lossPer_p.item()),
        )
        log(
            lf,
            "    velocity:  abs. difference, ratio: %f , %f "
            % (
                np.sum(np.abs(outputs_cpu[1] - targets_cpu[1]))
                + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2])),
                lossPer_v.item(),
            ),
        )
        log(
            lf,
            "    aggregate: abs. difference, ratio: %f , %f "
            % (np.sum(np.abs(outputs_cpu - targets_cpu)), lossPer.item()),
        )

        # Calculate the norm
        input_ndarray = inputs_cpu.cpu().numpy()[0]
        v_norm = (
            np.max(np.abs(input_ndarray[0, :, :])) ** 2
            + np.max(np.abs(input_ndarray[1, :, :])) ** 2
        ) ** 0.5

        outputs_denormalized = dataset.denormalize(outputs_cpu, v_norm)
        targets_denormalized = dataset.denormalize(targets_cpu, v_norm)

        # denormalized error
        outputs_denormalized_comp = np.array([outputs_denormalized])
        outputs_denormalized_comp = torch.from_numpy(outputs_denormalized_comp)
        targets_denormalized_comp = np.array([targets_denormalized])
        targets_denormalized_comp = torch.from_numpy(targets_denormalized_comp)

        targets_denormalized_comp, outputs_denormalized_comp = (
            targets_denormalized_comp.float(),
            outputs_denormalized_comp.float(),
        )

        outputs_dn.data.resize_as_(outputs_denormalized_comp).copy_(
            outputs_denormalized_comp
        )
        targets_dn.data.resize_as_(targets_denormalized_comp).copy_(
            targets_denormalized_comp
        )

        if save_outputs:
            np.savez_compressed(
                "./" + prefix + "_results_test_" + suffix + s + "/" + test_files[i],
                a=outputs_dn,
            )

        lossL1_dn = criterionL1(outputs_dn, targets_dn)
        L1val_dn_accum += lossL1_dn.item()

        # write output image, note - this is currently overwritten for multiple models
        os.chdir("./" + prefix + "_results_test_" + suffix + s + "/")
        
        utils.imageOut2(
            test_files[i][:-4],
            outputs_cpu,
            targets_cpu,
            input_ndarray,
        )
        os.chdir("../")

    log(lf, "\n")
    L1val_accum /= len(testLoader)
    lossPer_p_accum /= len(testLoader)
    lossPer_v_accum /= len(testLoader)
    lossPer_accum /= len(testLoader)
    L1val_dn_accum /= len(testLoader)
    log(
        lf,
        "Loss percentage (p, v, combined): %f %%    %f %%    %f %% "
        % (lossPer_p_accum * 100, lossPer_v_accum * 100, lossPer_accum * 100),
    )
    log(lf, "L1 error: %f" % (L1val_accum))
    log(lf, "Denormalized error: %f" % (L1val_dn_accum))
    log(lf, "\n")

    avgLoss += lossPer_accum
    losses.append(lossPer_accum)

if len(losses) > 1:
    avgLoss /= len(losses)
    lossStdErr = np.std(losses) / math.sqrt(len(losses))
    log(
        lf,
        "Averaged relative error and std dev across models:   %f , %f "
        % (avgLoss, lossStdErr),
    )


end_time = time.time()
time_taken = (end_time - start_time) / 60  # minutes
log(lf, "\nTime taken for testing, {} minutes".format(time_taken))

