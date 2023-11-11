"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import imp

import numpy as np
import torch as th
import torch.distributed as dist
import pickle

import os, sys
sys.path.insert(1, os.getcwd()) 


from diffusion_openai.video_datasets import load_data
from diffusion_openai import dist_util, logger
from diffusion_openai.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)



def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    cond_kwargs = {}
    cond_len = 0
    cond_frames = []
    # get data to condition on
    if args.cond_generation:
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            deterministic=False,
            rgb=args.rgb
        )
        num = ""
        for i in args.cond_frames:
            if i == ",":
                cond_frames.append(int(num))
                num = ""
            else:
                num = num + i
        # cond_frames = list(int(i) for i in args.cond_frames if i != ",")
        ref_frames = list(i for i in range(args.seq_len) if i not in cond_frames)
        logger.log(f"cond_frames: {cond_frames}")
        logger.log(f"ref_frames: {ref_frames}")
        logger.log(f"seq_len: {args.seq_len}")
        cond_len = len(cond_frames)
    cond_kwargs["resampling_steps"] = args.resample_steps
    cond_kwargs["cond_frames"] = cond_frames

    if args.rgb:
        channels = 3
    else:
        channels = 1

    logger.log("sampling...")
    all_videos = []
    all_gt = []
    while len(all_videos) * args.batch_size < args.num_samples:
        
        sample = []
        if args.cond_generation:
            video, _ = next(data)
            cond_kwargs["cond_img"] = video[:,:,cond_frames].to(dist_util.dev()) 
            video = video.to(dist_util.dev())
            sample.append(cond_kwargs["cond_img"])
        

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        sample_len = 0
        while sample_len < args.seq_len:
            temp_len = args.prediction_step + cond_len
            print("seq_len:", temp_len)
            print("cond_frames:", cond_kwargs["cond_frames"])
            temp_sample = sample_fn(
                model,
                (args.batch_size, channels, temp_len, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                progress=False,
                cond_kwargs=cond_kwargs
            )

            sample.append(temp_sample[:,:,-args.prediction_step:])
        
            if temp_sample.shape[2] > args.max_cond_frames:
                cond_kwargs["cond_img"] = temp_sample[:,:,-args.max_cond_frames:]
            else:
                cond_kwargs["cond_img"] = temp_sample
            cond_kwargs["cond_frames"] = [j for j in range(cond_kwargs["cond_img"].shape[2])]
            cond_len = len(cond_kwargs["cond_frames"])
            print("sample:", temp_sample.shape)
            sample_len = np.sum([vid.shape[2] for vid in sample])
            print("len: ",sample_len)
            
        
        sample = th.cat(sample, dim=2)
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 4, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_videos.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_videos) * args.batch_size} samples")

        if args.cond_generation and args.save_gt:

            video = ((video + 1) * 127.5).clamp(0, 255).to(th.uint8)
            video = video.permute(0, 2, 3, 4, 1)
            video = video.contiguous()

            gathered_videos = [th.zeros_like(video) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_videos, video)  # gather not supported with NCCL
            all_gt.extend([video.cpu().numpy() for video in gathered_videos])
            logger.log(f"created {len(all_gt) * args.batch_size} videos")


    arr = np.concatenate(all_videos, axis=0)

    if args.cond_generation and args.save_gt:
        arr_gt = np.concatenate(all_gt, axis=0)


    if dist.get_rank() == 0:

        shape_str = "x".join([str(x) for x in arr.shape])
        name = f"{shape_str}_{len(args.cond_frames)}"
        logger.log(f"saving samples to {os.path.join(logger.get_dir(), name)}")
        np.savez(os.path.join(logger.get_dir(), shape_str), arr)

        if args.cond_generation and args.save_gt:
            shape_str_gt = "x".join([str(x) for x in arr_gt.shape])
            name_gt = f"{shape_str}_{len(args.cond_frames)}"
            logger.log(f"saving ground_truth to {os.path.join(logger.get_dir(), name_gt)}")
            np.savez(os.path.join(logger.get_dir(), shape_str_gt), arr_gt)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=10,
        use_ddim=False,
        model_path="",
        save_gif=False, 
        seq_len=30,
        sampling_type="generation",
        cond_frames="0,",
        cond_generation=False,
        resample_steps=1,
        data_dir='',
        save_gt=False,
        prediction_step=10,
        seed = 17,
        max_cond_frames=5
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
