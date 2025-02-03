# Deep-Flow-HighWake 

This repository contains code that is used in our study titled, “Addressing performance improvement of a neural network model for Reynolds-averaged Navier–Stokes solutions with high wake formation,” by A. A. Kumar and A. Assam published in Engineering Computations, 2024 (https://doi.org/10.1108/EC-08-2023-0446).


This repository was forked from <https://github.com/thunil/Deep-Flow-Prediction>, which was used in the work titled “Deep learning methods for reynolds-averaged navier–stokes simulations of airfoil flows,” by N. Thuerey, K. Weißenow, L. Prantl, and X. Hu, published in AIAA Journal, 2019 (https://doi.org/10.2514/1.J058291).

Code for data generation using multiprocessing, and network training and evaluation is provided.

Linux OS or WSL installation is required for the use of OpenFOAM and Gmsh to generate data.

## Software Information

The following software and their versions were used in this study.
```
OpenFOAM-5.x
Gmsh 3.0.6
pytorch 1.12.0
```

## Instructions

Instructions are given for Ubuntu.

### Get airfoils from UIUC database

Change directory to `data` and execute `download_airfoils.sh`. Some of the downloaded airfoils will be split into a smaller directory with 30 airfoils. This smaller directory is only used for testing models and not used for training models.

### Generate Training data

1. Change directory to `data`.
2. Set the required number of samples and number of cores available in `dataGen.py`. Parallel generation of training data (multi_processing) is enabled by default.
3. Run the code using the command ` python dataGen.py`. By default sample will be saved in `./train`.

### Training the model

1. Change directory to `train`.
2. Set the required training parameters in the `runTrainCpu.py`. By default the script trains a model with 147,555 trainable parameters using 10 samples from `../data/train` for 1000 iterations.
3. The model starts training. The training and validation errors during training will be displayed along with epoch number. At the end of training, the model will be saved as `_modelG_`.

### Testing the model

1. In the `train` directory, run the command `python runTestCpu.py` to test the models saved in `train` directory.
2. Make sure that the `expo` parameter, for the size of the model, is the same as which were used in the training script to train the model.
3. Inference results and test results are saved at the end.
