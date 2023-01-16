# Diffusion Models
This is a simple implementation of diffusion models for downscaling climate data. It is based on the implementation of diffusion models from [Dominic Rampas](https://github.com/dome272/Diffusion-Models-pytorch), and closely follows Algorithm 1 from the [DDPM](https://arxiv.org/pdf/2006.11239.pdf) paper. So far, it can be applied to both wind as well as temperature data.

## Overview on files
- ddpm_downscale.py: Main file used for the diffusion class and training the diffusion model
- modules.py: Helper file that includes the components and the UNet model used
- utils.py: Helper file for dataset classes, data loading and plotting / saving images
- metrics.py: Metrics for evaluation
- generate_testsamples.py: File that was used for generating samples from the test data
- data_permutation: Permutation of the images that defines the split into train and test data for the final results. Will be loaded automatically if the dataset_size is set to 10000.
- data.ipnyb: Jupyter notebook to download and save the dataset
- Plotting.ipnyb: Helper notebook to plot result plots

## Data
You can download the wind data from [Google Drive](https://drive.google.com/file/d/1zLTmCfsZIl0Sb8FPS9oYPME0HOuFSt8u/view?usp=share_link)

## Overview on hyperparameters
- batch_size: batch size; default 15
- dataset_size: maximum size of dataset for both train and test images together; default is 10000 and the images are split into 5000 train and 5000 test images automatically using the data_permutation file; parameter can be set to smaller values to enable faster Training
- noise_schedule: noise schedule in diffsion model, can be linear or cosine
- epochs: number of epochs to run
- lr: learning rate
- dataset_type: wind, temperature or MNIST
- repeat_observations: if set > 0, repeats each image in the dataset multiple times; can be used for simple test experiments, e.g. to see if the model can overfit a single image; not relevant for actual runs; default 1
- cfg_proportion: proportion of training steps that are done unconditionally, i.e. be setting the LR input to zero - a value around 0.1 or 0.2  has shown to enhance performance; default 0
- image_size: size of the HR image; default 64; can be set to smaller values to enable faster training
- shuffle: defines the shuffle argument in pytorch's dataloader; True or False; default True
- resolution_ratio: ratio of resolution HR image / LR image; default 4
- folder_prefix: prefix of folder name in which results should be saved

## Repeat experiments on wind data
### Training
1. (optional) Configure Hyperparameters in ```ddpm_downscale.py```; most hyperparameters can also be set via the command line (see below)
2. Set path to dataset in ```ddpm_downscale.py```
3. ```python ddpm_downscale.py```, optionally with more hyperparameters, e.g. ```python ddpm_downscale.py --image_size 32```

### Command to reproduce results
The results were generated as follows:
- ```python ddpm_downscale.py --batch_size 15 --dataset_size 10000 --lr 0.0001 --noise_schedule linear --epochs 500 --dataset_type wind --cfg_proportion 0.1 --image_size 64 --shuffle True --resolution_ratio 4```
- ```python ddpm_downscale.py --batch_size 15 --dataset_size 10000 --lr 0.0001 --noise_schedule linear --epochs 500 --dataset_type wind --cfg_proportion 0.1 --image_size 64 --shuffle True --resolution_ratio 8```
- ```python ddpm_downscale.py --batch_size 15 --dataset_size 10000 --lr 0.0001 --noise_schedule linear --epochs 500 --dataset_type wind --cfg_proportion 0.1 --image_size 64 --shuffle True --resolution_ratio 16```

### Sampling
1. Download the checkpoints for the models [here](https://drive.google.com/drive/folders/18J1K70g4nIK_8SViRdr7kYJc7C0MdUrH?usp=share_link).
- Choose DDPM_downscale_v2_fixeddata for the main model with a resolution ratio of 4. It performs downscaling from 16x16 to 64x64 images.
- Choose DDPM_downscale_v3_fixeddata_resratio8 or DDPM_downscale_v4_fixeddata_resratio16 for the models with higher resolution ratio. The former performs downscaling for 8x8 -> 64x64 and the latter 4x4 -> 64x64.
2. Set path to dataset in ```generate_testsamples.py```, optionally more hyperparameters.
3. Generate testsamples with ```generate_testsamples.py```, e.g.
```"python generate_testsamples.py --batch_size 4 --dataset_size 10000 --dataset_type wind --image_size 64 --shuffle True --folder_name v2_fixeddata```.
<hr>
