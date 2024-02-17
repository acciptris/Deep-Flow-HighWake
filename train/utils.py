import math
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image


# add line to logfiles
def log(file, line, doPrint=True):
    f = open(file, "a+")
    f.write(line + "\n")
    f.close()
    if doPrint:
        print(line)


# reset log file
def resetLog(file):
    f = open(file, "w")
    f.close()


# compute learning rate with decay in second half
def computeLR(i, epochs, minLR, maxLR):
    if i < epochs * 0.5:
        return maxLR
    e = (i / float(epochs) - 0.5) * 2.0
    # rescale second half to min/max range
    fmin = 0.0
    fmax = 6.0
    e = fmin + e * (fmax - fmin)
    f = math.pow(0.5, e)
    return minLR + (maxLR - minLR) * f


# image output
def imageOut(
    filename, _outputs, _targets, saveTargets=False, normalize=False, saveMontage=True
):
    outputs = np.copy(_outputs)
    targets = np.copy(_targets)

    s = outputs.shape[1]  # should be 128
    if saveMontage:
        new_im = Image.new("RGB", ((s + 10) * 3, s * 2), color=(255, 255, 255))
        BW_im = Image.new("RGB", ((s + 10) * 3, s * 3), color=(255, 255, 255))

    for i in range(3):
        outputs[i] = np.flipud(outputs[i].transpose())
        targets[i] = np.flipud(targets[i].transpose())
        min_value = min(np.min(outputs[i]), np.min(targets[i]))
        max_value = max(np.max(outputs[i]), np.max(targets[i]))
        if normalize:
            outputs[i] -= min_value
            targets[i] -= min_value
            max_value -= min_value
            outputs[i] /= max_value
            targets[i] /= max_value
        else:  # from -1,1 to 0,1
            outputs[i] -= -1.0
            targets[i] -= -1.0
            outputs[i] /= 2.0
            targets[i] /= 2.0

        if not saveMontage:
            suffix = ""
            if i == 0:
                suffix = "_pressure"
            elif i == 1:
                suffix = "_velX"
            else:
                suffix = "_velY"

            im = Image.fromarray(cm.magma(outputs[i], bytes=True))
            im = im.resize((512, 512))
            im.save(filename + suffix + "_pred.png")

            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            if saveTargets:
                im = im.resize((512, 512))
                im.save(filename + suffix + "_target.png")

        if saveMontage:
            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            new_im.paste(im, ((s + 10) * i, s * 0))
            im = Image.fromarray(cm.magma(outputs[i], bytes=True))
            new_im.paste(im, ((s + 10) * i, s * 1))

            im = Image.fromarray(targets[i] * 256.0)
            BW_im.paste(im, ((s + 10) * i, s * 0))
            im = Image.fromarray(outputs[i] * 256.0)
            BW_im.paste(im, ((s + 10) * i, s * 1))
            imE = Image.fromarray(np.abs(targets[i] - outputs[i]) * 10.0 * 256.0)
            BW_im.paste(imE, ((s + 10) * i, s * 2))

    if saveMontage:
        new_im.save(filename + ".png")
        BW_im.save(filename + "_bw.png")


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


c_transparent = matplotlib.colors.colorConverter.to_rgba("white", alpha=0)
c_black = matplotlib.colors.colorConverter.to_rgba("black")
cmap_black_transparent = matplotlib.colors.LinearSegmentedColormap.from_list(
    "cmap_black_transparent", [c_transparent, c_black], 2
)


def imageOut2(
    filename,
    _outputs,
    _targets,
    _inputs,
    max_targets=[1, 1, 1],
    v_norm=[1],
    scale=2,
    streamplot_density=2,
    streamplot_linewidth=1,
    streamplot_arrowsize=0.5,
):
    outputs = np.copy(_outputs)
    targets = np.copy(_targets)
    inputs = np.copy(_inputs)

    s = outputs.shape[1]  # should be 128
    s *= scale
    new_im = Image.new("RGB", ((s + 10) * 4, s * 3), color=(255, 255, 255))

    for i in range(3):
        outputs[i] = np.flipud(outputs[i].transpose())
        targets[i] = np.flipud(targets[i].transpose())

        # from -1,1 to 0,1
        outputs[i] -= -1.0
        targets[i] -= -1.0
        outputs[i] /= 2.0
        targets[i] /= 2.0

        im = Image.fromarray(cm.magma(targets[i], bytes=True))
        im = im.resize((s, s))
        new_im.paste(im, ((s + 10) * i, s * 0))
        im = Image.fromarray(cm.magma(outputs[i], bytes=True))
        im = im.resize((s, s))
        new_im.paste(im, ((s + 10) * i, s * 1))
        im = Image.fromarray(np.abs(targets[i] - outputs[i]) * 10.0 * 256.0)
        im = im.resize((s, s))
        new_im.paste(im, ((s + 10) * i, s * 2))

    outputs = np.copy(_outputs)
    targets = np.copy(_targets)
    x, y = np.meshgrid(np.arange(0, s / scale), np.arange(0, s / scale))
    u_true = targets[1].transpose()  # * max_targets[1] * v_norm
    v_true = targets[2].transpose()  # * max_targets[2] * v_norm
    u_pred = outputs[1].transpose()  # * max_targets[1] * v_norm
    v_pred = outputs[2].transpose()  # * max_targets[2] * v_norm
    mask = inputs[2].transpose()

    fig = plt.figure(figsize=(s / 100, s / 100))
    fig.tight_layout(pad=0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.contourf(mask, cmap=cmap_black_transparent, zorder=100)
    ax.streamplot(
        x,
        y,
        u_true,
        v_true,
        density=streamplot_density,
        linewidth=streamplot_linewidth,
        arrowsize=streamplot_arrowsize,
        zorder=50,
    )
    true_stream_im = fig2img(fig)
    new_im.paste(true_stream_im, ((s + 10) * 3, s * 0))

    fig = plt.figure(figsize=(s / 100, s / 100))
    fig.tight_layout(pad=0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.contourf(mask, cmap=cmap_black_transparent, zorder=100)
    ax.streamplot(
        x,
        y,
        u_pred,
        v_pred,
        density=streamplot_density,
        linewidth=streamplot_linewidth,
        arrowsize=streamplot_arrowsize,
        zorder=50,
    )
    pred_stream_im = fig2img(fig)
    new_im.paste(pred_stream_im, ((s + 10) * 3, s * 2))

    new_im.save(filename + "_stream.png")


# save single image
def saveAsImage(filename, field_param):
    field = np.copy(field_param)
    field = np.flipud(field.transpose())

    min_value = np.min(field)
    max_value = np.max(field)
    field -= min_value
    max_value -= min_value
    field /= max_value

    im = Image.fromarray(cm.magma(field, bytes=True))
    im = im.resize((512, 512))
    im.save(filename)


# read data split from command line
def readProportions():
    flag = True
    while flag:
        input_proportions = input(
            "Enter total numer for training files and proportions for training (normal, superimposed, sheared respectively) seperated by a comma such that they add up to 1: "
        )
        input_p = input_proportions.split(",")
        prop = [float(x) for x in input_p]
        if prop[1] + prop[2] + prop[3] == 1:
            flag = False
        else:
            print("Error: poportions don't sum to 1")
            print("##################################")
    return prop


# helper from data/utils
def makeDirs(directoryList):
    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)


def replaceLine(fileName, lineNum, lineContent):
    # lineNum is 1 index based
    with open(fileName, "r") as file:
        lines = file.readlines()

    lines[lineNum - 1] = lineContent

    with open(fileName, "w") as file:
        file.writelines(lines)
