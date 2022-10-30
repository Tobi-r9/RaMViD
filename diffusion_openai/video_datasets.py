from random import sample
from PIL import Image, ImageSequence
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import av

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, rgb=True, seq_len=20
):
    """
    For a dataset, create a generator over (videos, kwargs) pairs.

    Each video is an NCLHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which frames are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_video_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    
    entry = all_files[0].split(".")[-1]
    if entry in ["avi", "mp4"]:
        dataset = VideoDataset_mp4(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            rgb=rgb,
            seq_len=seq_len
        )
    elif entry in ["gif"]:
        dataset = VideoDataset_gif(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            rgb=rgb,
            seq_len=seq_len
        )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True
        )
    while True:
        yield from loader


def _list_video_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["gif", "avi", "mp4"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_video_files_recursively(full_path))
    return results

class VideoDataset_mp4(Dataset):
    def __init__(self, resolution, video_paths, classes=None, shard=0, num_shards=1, rgb=True, seq_len=20):
        super().__init__()
        self.resolution = resolution
        self.local_videos = video_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.rgb = rgb
        self.seq_len = seq_len

    def __len__(self):
        return len(self.local_videos)

    def __getitem__(self, idx):
        path = self.local_videos[idx]
        arr_list = []
        video_container = av.open(path)
        n = video_container.streams.video[0].frames
        frames = [i for i in range(n)]
        if n > self.seq_len:
            start = np.random.randint(0, n-self.seq_len)
            frames = frames[start:start + self.seq_len]
        for id, frame_av in enumerate(video_container.decode(video=0)):
        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
            if (id not in frames):
                continue
            frame = frame_av.to_image()
            while min(*frame.size) >= 2 * self.resolution:
                frame = frame.resize(
                    tuple(x // 2 for x in frame.size), resample=Image.BOX
                )
            scale = self.resolution / min(*frame.size)
            frame =frame.resize(
                tuple(round(x * scale) for x in frame.size), resample=Image.BICUBIC
            )

            if self.rgb:
                arr = np.array(frame.convert("RGB"))
            else:
                arr = np.array(frame.convert("L"))
                arr = np.expand_dims(arr, axis=2)
            crop_y = (arr.shape[0] - self.resolution) // 2
            crop_x = (arr.shape[1] - self.resolution) // 2
            arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
            arr = arr.astype(np.float32) / 127.5 - 1
            arr_list.append(arr)
        arr_seq = np.array(arr_list)
        arr_seq = np.transpose(arr_seq, [3, 0, 1, 2])
        # fill in missing frames with 0s
        if arr_seq.shape[1] < self.seq_len:
            required_dim = self.seq_len - arr_seq.shape[1]
            fill = np.zeros((3, required_dim, self.resolution, self.resolution))
            arr_seq = np.concatenate((arr_seq, fill), axis=1)
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return arr_seq, out_dict

class VideoDataset_gif(Dataset):
    def __init__(self, resolution, video_paths, classes=None, shard=0, num_shards=1, rgb=True, seq_len=20):
        super().__init__()
        self.resolution = resolution
        self.local_videos = video_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.rgb = rgb
        self.seq_len = seq_len

    def __len__(self):
        return len(self.local_videos)

    def __getitem__(self, idx):
        path = self.local_videos[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_videos = Image.open(f)
            arr_list = []
            for frame in ImageSequence.Iterator(pil_videos):

            # We are not on a new enough PIL to support the `reducing_gap`
            # argument, which uses BOX downsampling at powers of two first.
            # Thus, we do it by hand to improve downsample quality.
                while min(*frame.size) >= 2 * self.resolution:
                    frame = frame.resize(
                        tuple(x // 2 for x in frame.size), resample=Image.BOX
                    )
                scale = self.resolution / min(*frame.size)
                frame =frame.resize(
                    tuple(round(x * scale) for x in frame.size), resample=Image.BICUBIC
                )

                if self.rgb:
                    arr = np.array(frame.convert("RGB"))
                else:
                    arr = np.array(frame.convert("L"))
                    arr = np.expand_dims(arr, axis=2)
                crop_y = (arr.shape[0] - self.resolution) // 2
                crop_x = (arr.shape[1] - self.resolution) // 2
                arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
                arr = arr.astype(np.float32) / 127.5 - 1
                arr_list.append(arr)
        arr_seq = np.array(arr_list)
        arr_seq = np.transpose(arr_seq, [3, 0, 1, 2])
        if arr_seq.shape[1] > self.seq_len:
            start = np.random.randint(0, arr_seq.shape[1]-self.seq_len)
            arr_seq = arr_seq[:,start:start + self.seq_len]
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return arr_seq, out_dict
