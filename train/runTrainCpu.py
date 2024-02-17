import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import dataset
import utils
from DfpNet import TurbNetG, weights_init

######## Settings ########

# number of training iterations
iterations = 1000
# batch size
batch_size = 1
# learning rate, generator
lrG = 0.0004
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 3
# data set config
# prop: list, whose first element is the number of samples to be randomly chosen from train_directory
prop = [10]
# prop=[1000,0.75,0,0.25] # mix data from multiple directories

# save txt files with per epoch loss?
saveL1 = True

train_directory = "../data/train/"

prefix = ""
suffix = ""

if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

dropout = 0.01  # note, the original runs from https://arxiv.org/abs/1810.08217 used slight dropout, but the effect is minimal; conv layers "shouldn't need" dropout, hence set to 0 here.
doLoad = ""  # optional, path to pre-trained model

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))
print("Dropout: {}".format(dropout))

##########################

start_time = time.time()

torch.set_num_threads(64)
seed = random.randint(0, 2 ** 32 - 1)
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# create pytorch data object with dfp dataset
print("Loadin Data...")
data = dataset.TurbDataset(dataProp=prop, shuffle=1, dataDir=train_directory)
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))
dataValidation = dataset.ValiDataset(data)
valiLoader = DataLoader(
    dataValidation, batch_size=batch_size, shuffle=False, drop_last=True
)
print("Validation batches: {}".format(len(valiLoader)))

# setup training
epochs = int(iterations / len(trainLoader) + 0.5)
netG = TurbNetG(channelExponent=expo, dropout=dropout)
# print(netG) # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized TurbNet with {} trainable params ".format(params))

netG.apply(weights_init)
if len(doLoad) > 0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model " + doLoad)

criterionL1 = nn.L1Loss()
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

targets = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
inputs = Variable(torch.FloatTensor(batch_size, 3, 128, 128))

##########################
with open(prefix + "_modelG_" + suffix + "_parameters.txt", "w") as f:
    print("Using {} threads".format(torch.get_num_threads()), file=f)
    print("Random seed: {}".format(seed), file=f)
    print("Train Directory: {}".format(train_directory), file=f)
    print("Batch size: {}".format(batch_size), file=f)
    print("Prop: {}".format(prop), file=f)
    print("LR: {}".format(lrG), file=f)
    print("LR decay: {}".format(decayLr), file=f)
    print("Iterations: {}".format(iterations), file=f)
    print("Epochs: {}".format(epochs), file=f)
    print("Dropout: {}".format(dropout), file=f)
    print("Training samples: {}".format(len(data)), file=f)
    print("Validation Samples: {}".format(len(dataValidation)), file=f)
    print("Training batches: {}".format(len(trainLoader)), file=f)
    print("Validation batches: {}".format(len(valiLoader)), file=f)
    print("Expo: {}".format(expo), file=f)
    print("Initialized TurbNet with {} trainable params ".format(params), file=f)
    if len(doLoad) > 0:
        print("Loaded model " + doLoad, file=f)

avg_epoch_time = 0
for epoch in range(epochs):
    epoch_start_time = time.time()

    print("Starting epoch {} / {}".format((epoch + 1), epochs))

    netG.train()
    L1_accum = 0.0
    for i, traindata in enumerate(trainLoader, 0):
        inputs_cpu, targets_cpu = traindata
        inputs.data.copy_(inputs_cpu.float())
        targets.data.copy_(targets_cpu.float())

        # compute LR decay
        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG * 0.1, lrG)
            if currLr < lrG:
                for g in optimizerG.param_groups:
                    g["lr"] = currLr

        optimizerG.zero_grad()
        gen_out = netG(inputs)

        lossL1 = criterionL1(gen_out, targets)
        lossL1.backward()

        optimizerG.step()

        lossL1viz = lossL1.item()
        L1_accum += lossL1viz

        if i == len(trainLoader) - 1:
            logline = "Epoch: {}, batch-idx: {}, L1: {}".format(epoch + 1, i, lossL1viz)
            print(logline)

    # validation
    netG.eval()
    L1val_accum = 0.0
    for i, validata in enumerate(valiLoader, 0):
        inputs_cpu, targets_cpu = validata
        inputs.data.copy_(inputs_cpu.float())
        targets.data.copy_(targets_cpu.float())

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()

        lossL1 = criterionL1(outputs, targets)
        L1val_accum += lossL1.item()

        if i == 0:
            input_ndarray = inputs_cpu.cpu().numpy()[0]
            v_norm = (
                np.max(np.abs(input_ndarray[0, :, :])) ** 2
                + np.max(np.abs(input_ndarray[1, :, :])) ** 2
            ) ** 0.5

            outputs_denormalized = data.denormalize(outputs_cpu[0], v_norm)
            targets_denormalized = data.denormalize(
                targets_cpu.cpu().numpy()[0], v_norm
            )
            utils.makeDirs([prefix + "_results_train_" + suffix])
            utils.imageOut(
                prefix + "_results_train_" + suffix + "/epoch{}_{}".format(epoch, i),
                outputs_denormalized,
                targets_denormalized,
                saveTargets=True,
            )

    # data for graph plotting
    L1_accum /= len(trainLoader)
    L1val_accum /= len(valiLoader)

    epoch_end_time = time.time()
    epoch_time_taken = epoch_end_time - epoch_start_time
    avg_epoch_time += epoch_time_taken

    if saveL1:
        if epoch == 0:
            utils.resetLog(prefix + suffix + "_L1.txt")
            utils.resetLog(prefix + suffix + "_L1val.txt")
            utils.resetLog(prefix + suffix + "_epochTime.txt")
        utils.log(prefix + suffix + "_L1.txt", "{} ".format(L1_accum), False)
        utils.log(prefix + suffix + "_L1val.txt", "{} ".format(L1val_accum), False)
        utils.log(
            prefix + suffix + "_epochTime.txt", "{} ".format(epoch_time_taken), False
        )

    print("Validation Loss: {}".format(L1val_accum))
    print(
        "ETC: {}\n".format(
            datetime.fromtimestamp(
                epoch_start_time + (epochs - epoch - 1) * epoch_time_taken
            )
        )
    )

avg_epoch_time /= epochs
torch.save(netG.state_dict(), prefix + "_modelG_" + suffix)


end_time = time.time()
time_taken = (end_time - start_time) / 60  # minutes
print("\nTime taken for training, {} minutes".format(time_taken))
with open(prefix + "_modelG_" + suffix + "_parameters.txt", "a") as f:
    print("\nTime taken for training, {} minutes".format(time_taken), file=f)
    print("Average epoch time, {} seconds".format(avg_epoch_time), file=f)

