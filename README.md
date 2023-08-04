# RaMViD

Official code for the paper:

> [Diffusion Models for Video Prediction and Infilling](https://arxiv.org/abs/2206.07696) <br/>
> Tobias Höppe, Arash Mehrjou, Stefan Bauer, Didrik Nielsen, Andrea Dittadi <br/>
> TMLR, 2022

**Project website: https://sites.google.com/view/video-diffusion-prediction**

This code and README are based on https://github.com/openai/improved-diffusion

## Installation

Import and create the enroot container 
```
$ enroot import docker://nvcr.io#nvidia/pytorch:21.04-py3
$ enroot create --name container_name nvidia+pytorch+21.04-py3.sqsh
```
and run in addition
```
pip install torch
pip install tqdm
pip install blobfile>=0.11.0
pip install mpi4py
pip install matplotlib
pip install av 
```

## Preparing Data

Our dataloader can handle videos in the .gif, .mp4 or .av format. Create a folder with your data and simply pass `--data_dir path/to/videos` to the training script

## Training

Similar to the original code baes, will split up our hyperparameters into three groups: model architecture, diffusion process, and training flags. 

Kinetics-600:  
```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --scale_time_dim 0";
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear";
TRAIN_FLAGS="--lr 2e-5 --batch_size 8 --microbatch 2 --seq_len 16 --max_num_mask_frames 4 --uncondition_rate 0.25";
```

BAIR:  
```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 2 --scale_time_dim 0";
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear";
TRAIN_FLAGS="--lr 2e-5 --batch_size 4 --microbatch 2 --seq_len 20 --max_num_mask_frames 4 --uncondition_rate 0.25";
```

UCF-101:  
```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --scale_time_dim 0";
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear";
TRAIN_FLAGS="--lr 2e-5 --batch_size 8 --microbatch 2 --seq_len 16 --max_num_mask_frames 4 --uncondition_rate 0.75";
```

Once you have setup your hyper-parameters, you can run an experiment like so:

```
python scripts/video_train.py --data_dir path/to/videos $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

You may also want to train in a distributed manner. In this case, run the same command with `mpirun`:

```
mpirun -n $NUM_GPUS python scripts/video_train.py --data_dir path/to/videos $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

if you want to continue training a model, you should add the model path to the command via `--resume_checkpoint`. In the same folder you should have the saved optimizer and ema model.

Pre-trained models:<br/>
[Model trained on Kinetics-600 (K=4, p=0.25)](https://1drv.ms/f/s!Amih97wvmSyWgosH_uJoN-BsH_RWkw?e=P0Wg8n)  <br/>
[Model trained on BAIR (K=4, p=0.25)](https://1drv.ms/f/s!Amih97wvmSyWgosGv9ekMoXGy_6CSg?e=ElIg3i)  <br/>
[Model trained on UCF-101 (K=4, p=0.75)](https://1drv.ms/f/s!Amih97wvmSyWgosIEgaDNoDbxRFDYQ?e=PWZnNA)  <br/>

When training in a distributed manner, you must manually divide the `--batch_size` argument by the number of ranks. In lieu of distributed training, you may use `--microbatch 16` (or `--microbatch 1` in extreme memory-limited cases) to reduce memory usage.

The logs and saved models will be written to a logging directory determined by the `OPENAI_LOGDIR` environment variable. If it is not set, then a temporary directory will be created in `/tmp`.

## Sampling

The above training script saves checkpoints to `.pt` files in the logging directory. These checkpoints will have names like `ema_0.9999_200000.pt` and `model200000.pt`. You will likely want to sample from the EMA models, since those produce much better samples.

Once you have a path to your model, you can generate a large batch of samples like so:

```
python scripts/video_sample.py --model_path /path/to/model.pt $MODEL_FLAGS $DIFFUSION_FLAGS
```

Again, this will save results to a logging directory. Samples are saved as a large `npz` file, where `arr_0` in the file is a large batch of samples.

Just like for training, you can run `image_sample.py` through MPI to use multiple GPUs and machines.

You can change the number of sampling steps using the `--timestep_respacing` argument. For example, `--timestep_respacing 250` uses 250 steps to sample.



