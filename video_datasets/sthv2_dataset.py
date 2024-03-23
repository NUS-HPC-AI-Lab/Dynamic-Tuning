import os, sys
from typing import Optional, Tuple
import av
import io
import numpy as np

import torch
from torchvision import transforms
import decord
from .transform import (
    create_random_augment,
    random_resized_crop,
    random_short_side_scale_jitter,
    random_crop,
    )
from .random_erasing import RandomErasing
try:
  from .load_binary_internal import load_binary
except ImportError:
  from .load_binary import load_binary
  
import pandas as pd

class SthV2VideoDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        list_path: str,
        data_root: str,
        num_frames: int = 8,
        sampling_rate: int = 0,
        spatial_size: int = 224,
        num_spatial_views: int = 1,
        num_temporal_views: int = 1,
        random_sample: bool = True,
        mean: torch.Tensor = torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
        std: torch.Tensor = torch.Tensor([0.26862954, 0.26130258, 0.27577711]),
        auto_augment: Optional[str] = None,
        interpolation: str = 'bicubic',
        mirror: bool = False,
        load_labels: bool = True,
        resize_type: str = 'random_resized_crop',
        scale_range: Tuple[float, float] = (0.08, 1.0),
        random_erasing: Optional[RandomErasing] = None
    ):
        self.data_root = data_root
        self.interpolation = interpolation
        self.spatial_size = spatial_size
        self.load_labels = load_labels
        self.scale_range = scale_range
        self.random_erasing = random_erasing
        self.resize_type = resize_type
        # self.class_label = class_label

        if resize_type == 'random_resized_crop':
            assert 0. < scale_range[0] and scale_range[0] <= scale_range[1] and scale_range[1] <= 1.
        elif resize_type == 'random_short_side_scale_jitter':
            assert 1. <= scale_range[0] and scale_range[0] <= scale_range[1]
        else:
            assert False, 'resize type %s is not supported.' % resize_type

        self.mean, self.std = mean, std
        self.num_frames, self.sampling_rate = num_frames, sampling_rate

        if random_sample:
            assert num_spatial_views == 1 and num_temporal_views == 1
            self.random_sample = True
            self.mirror = mirror
            self.auto_augment = auto_augment
        else:
            assert auto_augment is None and not mirror
            self.random_sample = False
            self.num_temporal_views = num_temporal_views
            self.num_spatial_views = num_spatial_views

        with open(list_path) as f:
            self.data_list = f.read().splitlines()


    def __len__(self):
        return len(self.data_list)
    
    # @property
    # def classes(self):
    #     if "sthv2" not in self.class_label:
    #         classes_all = pd.read_csv(self.class_label)
    #         return classes_all.values.tolist()
        
    #     else:
    #         with open(self.class_label, "r") as f:
    #             classes_all = f.readlines()[1:]
    #         return [i.strip().split(",", maxsplit=1) for i in classes_all]
    

    def __getitem__(self, idx):
        line = self.data_list[idx]
        if self.load_labels:
            path, label = line.split(' ')
            label = int(label)
        else:
            path = line.split(' ')[0] # the list can be with or without labels
            label = None
        path = os.path.join(self.data_root, path)
        # raw_data = load_binary(path)
        
        # container = av.open(io.BytesIO(raw_data))
        # container.streams.video[0].thread_count = 1
        # frames = {}
        # for frame in container.decode(video=0):
        #     frames[frame.pts] = frame
        # container.close()
        # del container
        # frames = [frames[k] for k in sorted(frames.keys())]
        container = av.open(path)
        frames = {}
        for frame in container.decode(video=0):
            frames[frame.pts] = frame
        container.close()
        frames = [frames[k] for k in sorted(frames.keys())]
        total_frames = len(frames)
        
        
        if self.random_sample:
            frame_idx = self._random_sample_frame_idx(total_frames)
            frames = [frames[x].to_rgb().to_ndarray() for x in frame_idx]
            frames = torch.as_tensor(np.stack(frames)).float() / 255.

            if self.auto_augment is not None:
                aug_transform = create_random_augment(
                    input_size=(frames.size(1), frames.size(2)),
                    auto_augment=self.auto_augment,
                    interpolation=self.interpolation,
                )
                frames = frames.permute(0, 3, 1, 2) # T, C, H, W
                frames = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
                frames = aug_transform(frames)
                frames = torch.stack([transforms.ToTensor()(img) for img in frames])
                frames = frames.permute(0, 2, 3, 1)

            frames = (frames - self.mean) / self.std
            frames = frames.permute(3, 0, 1, 2) # C, T, H, W
            if self.resize_type == 'random_resized_crop':
                frames = random_resized_crop(
                    frames, self.spatial_size, self.spatial_size,
                    scale=self.scale_range,
                    interpolation=self.interpolation,
                )
            elif self.resize_type == 'random_short_side_scale_jitter':
                frames, _ = random_short_side_scale_jitter(
                    frames,
                    min_size=round(self.spatial_size * self.scale_range[0]),
                    max_size=round(self.spatial_size * self.scale_range[1]),
                    interpolation=self.interpolation,
                    )
                frames, _ = random_crop(frames, self.spatial_size)
            else:
                raise NotImplementedError()

            if self.random_erasing is not None:
                frames = self.random_erasing(frames.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

            if self.mirror and torch.rand(1).item() < 0.5:
                frames = frames.flip(dims=(-1,))
            
        else:
            # frame_idx = range(0, total_frames)
            # frame_dict = {idx: container[idx] for idx in np.unique(frame_idx)}
            # del container
            
            # frames = [frame_dict[x].asnumpy() for x in frame_idx]

            # frames = torch.as_tensor(np.stack(frames))
            # import time
            # start = time.time()
            # # frames = frames[:, :24, :, :]
            # frames = frames.float() / 255.
        
            # frames = (frames - self.mean) / self.std
            # frames = frames.permute(3, 0, 1, 2) # C, T, H, W
            # end = time.time()
            # print(end - start)
            temporal_crop_indices = self.get_temporal_crops_index(total_frames)
            clips = []
            for temporal_crop_indice in temporal_crop_indices:
                frames = [frames[x].to_rgb().to_ndarray() for x in temporal_crop_indice]
                # frame_dict = {idx: container[idx] for idx in np.unique(temporal_crop_indice)}
                # frames = [frame_dict[x].asnumpy() for x in temporal_crop_indice]

                frames = torch.as_tensor(np.stack(frames))
                frames = frames.float() / 255.
            
                frames = (frames - self.mean) / self.std
                frames = frames.permute(3, 0, 1, 2) # C, T, H, W
            
                if frames.size(-2) < frames.size(-1):
                    new_width = frames.size(-1) * self.spatial_size // frames.size(-2)
                    new_height = self.spatial_size
                else:
                    new_height = frames.size(-2) * self.spatial_size // frames.size(-1)
                    new_width = self.spatial_size
                
                clip = torch.nn.functional.interpolate(frames, size=(new_height, new_width),
                                mode=self.interpolation, align_corners=False)
                
                clips.append(clip)

            frames = sum([self._generate_spatial_crops(clip) for clip in clips], [])
            # end = time.time()
            # print(end - start)
            del container
            
            

            frames = torch.stack(frames)

            if self.random_erasing is not None:
                raise NotImplementedError('random erasing in non-random sample mode is'
                    'not supported.')

        if label is None:
            return frames
        else:
            return frames, label

    def get_temporal_crops_index(self, len_frames):
        if self.sampling_rate <= 0:
            assert self.num_temporal_views == 1, (
                'temporal multi-crop for uniform sampling is not supported.'
            )
            seg_size = (len_frames - 1) / self.num_frames
            frame_indices = []
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append((start + end) // 2)

            return [frame_indices]
        
        seg_len = (self.num_frames - 1) * self.sampling_rate + 1
        slide_len = max(len_frames - seg_len, 0)
        crop_indice = []
        for i in range(self.num_temporal_views):
            if self.num_temporal_views == 1:
                st = slide_len // 2
            else:
                st = round(slide_len / (self.num_temporal_views - 1) * i)
            crop_indice.append(list(map(lambda x: min(x + st, len_frames - 1), 
                                        range(0, self.num_frames * self.sampling_rate, self.sampling_rate))))
        return crop_indice
        
            
    def _generate_temporal_crops(self, frames):
        if self.sampling_rate <= 0:
            assert self.num_temporal_views == 1, (
                'temporal multi-crop for uniform sampling is not supported.'
            )
            seg_size = (frames.size(1) - 1) / self.num_frames
            frame_indices = []
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append((start + end) // 2)

            return [frames[:, frame_indices]]

        seg_len = (self.num_frames - 1) * self.sampling_rate + 1
        if frames.size(1) < seg_len:
            frames = torch.cat([frames, frames[:, -1:].repeat(1, seg_len - frames.size(1), 1, 1)], dim=1)
        slide_len = frames.size(1) - seg_len

        crops = []
        for i in range(self.num_temporal_views):
            if self.num_temporal_views == 1:
                st = slide_len // 2
            else:
                st = round(slide_len / (self.num_temporal_views - 1) * i)

            crops.append(frames[:, st: st + self.num_frames * self.sampling_rate: self.sampling_rate])
        
        return crops


    def _generate_spatial_crops(self, frames):
        if self.num_spatial_views == 1:
            assert min(frames.size(-2), frames.size(-1)) >= self.spatial_size
            h_st = (frames.size(-2) - self.spatial_size) // 2
            w_st = (frames.size(-1) - self.spatial_size) // 2
            h_ed, w_ed = h_st + self.spatial_size, w_st + self.spatial_size
            return [frames[:, :, h_st: h_ed, w_st: w_ed]]

        elif self.num_spatial_views == 3:
            assert min(frames.size(-2), frames.size(-1)) == self.spatial_size
            crops = []
            margin = max(frames.size(-2), frames.size(-1)) - self.spatial_size
            for st in (0, margin // 2, margin):
                ed = st + self.spatial_size
                if frames.size(-2) > frames.size(-1):
                    crops.append(frames[:, :, st: ed, :])
                else:
                    crops.append(frames[:, :, :, st: ed])
            return crops
        
        else:
            raise NotImplementedError()


    def _random_sample_frame_idx(self, len):
        frame_indices = []

        if self.sampling_rate <= 0: # tsn sample
            seg_size = (len - 1) / self.num_frames
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append(np.random.randint(start, end + 1)) # random
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= len:
            for i in range(self.num_frames):
                frame_indices.append(i * self.sampling_rate if i * self.sampling_rate < len else frame_indices[-1])
        else:
            start = np.random.randint(len - self.sampling_rate * (self.num_frames - 1))
            frame_indices = list(range(start, start + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, list_path: str, num_frames: int, num_views: int, spatial_size: int):
        with open(list_path) as f:
            self.len = len(f.read().splitlines())
        self.num_frames = num_frames
        self.num_views = num_views
        self.spatial_size = spatial_size

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        shape = [3, self.num_frames, self.spatial_size, self.spatial_size]
        if self.num_views != 1:
            shape = [self.num_views] + shape
        return torch.zeros(shape), 0
