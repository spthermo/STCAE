import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import torchvision.transforms.functional as F


def pil_transform(image, downscale, crop_dim):
    image = F.resize(image, image.size[1] // downscale)
    return F.center_crop(image, (crop_dim, crop_dim))

def pil_to_tensor(image):
    return F.to_tensor(image)

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img.convert('RGB')
            img = pil_transform(img, 2, 512)
            return pil_to_tensor(img)

def get_default_image_loader():
    return pil_loader

def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:03d}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

def get_class_labels(data):
    class_labels_map_temp = {}
    class_labels_map = {

    }
    index = 0
    for class_label in data:
        class_labels_map_temp[class_label] = index
        index += 1
    
    index = 0
    for idx in class_labels_map_temp:
        class_labels_map[idx] = index
        index += 1

    return class_labels_map

def get_video_names_and_annotations(data):
    video_names = []
    annotations = []
    num_frames = []
    gt_image_names = []

    for value in data['subjects']:
        name_splits = value['fullpath'].split('\\')
        video_names.append(name_splits[1] + '\\' + name_splits[2] + '\\' + name_splits[3])
        num_frames.append(len(value['frames']))
        annotations.append(value['label'])
        gt_image_names.append(value['gt_frame'])

    return video_names, num_frames, annotations, gt_image_names

def make_dataset(root_path, annotation_path, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, num_frames, annotations, gt_image_names = get_video_names_and_annotations(data)
    class_to_idx = get_class_labels(annotations)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        n_frames = int(num_frames[i])
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class

class SOR3D(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]['label']

        return clip, target

    def __len__(self):
        return len(self.data)
