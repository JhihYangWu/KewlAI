# written by JhihYang Wu <jhihyangwu@arizona.edu>

from torch.utils.data import Dataset
import random
import glob
import os
import imageio
import numpy as np
from animate_anyone.utils.utils import center_fill
import torch

class UBCFashionDataset_FirstPhase(Dataset):
    """
    Dataset class for UBC Fashion dataset for first phase of training.
    Look at README.md for information to download and add poses to it.
    """
    def __init__(self, path, distr="train"):
        """
        Constructor for UBC Fashion dataset class.

        Args:
            path (str): path to UBC Fashion dataset dir containing all the scenes.
            distr (str): which distribution of dataset to load (train, test)
        """
        assert distr in ["train", "test"]

        self.base_path = os.path.join(path, distr, "scenes")
        self.distr = distr
        self.ids = [scene_id for scene_id in os.listdir(self.base_path) if scene_id.isdigit()]

        self.resolution = 512

    def __len__(self):
        """
        Returns number of scenes in this dataset distribution.
        """
        return len(self.ids)
    
    def __getitem__(self, index):
        """
        First phase of training is on single images. This method will just return a random single frame of a scene.

        Args:
            index (int): index of scene to get.

        Returns:
            ref_images (tensor): rgb image .shape=(1, self.resolution, self.resolution, 3)
            expected_images (tensor): rgb image of the expected image .shape=(1, self.resolution, self.resolution, 3)
            expected_poses (tensor): pose image of the expected image .shape=(1, self.resolution, self.resolution, 3)
        """
        scene_id = self.ids[index]
        scene_path = os.path.join(self.base_path, scene_id)
        image_filenames = sorted(glob.glob(os.path.join(scene_path, "images", "*.png")))
        # randomly choose one image as reference
        ref_filename = random.choice(image_filenames)
        # randomly choose another image as expected and corresponding pose image
        expected_filename = random.choice(image_filenames)
        expected_pose_filename = os.path.join(scene_path, "dwposes", os.path.basename(expected_filename))

        # load images
        def load_img(path, fill_color=(255, 255, 255)):
            img = imageio.imread(path)
            # first three channels
            img = img[..., :3]
            # center fill to self.resolution x self.resolution
            img = center_fill(img, self.resolution, color=fill_color)
            # normalize
            img = img / 255.0
            img = 2.0 * img - 1.0  # so that it is in range [-1, 1]
            # convert to tensor
            img = torch.from_numpy(img)
            return img

        return {
            "ref_images": load_img(ref_filename).unsqueeze(0),
            "expected_images": load_img(expected_filename).unsqueeze(0),
            "expected_poses": load_img(expected_pose_filename, fill_color=(0, 0, 0)).unsqueeze(0),
        }

class UBCFashionDataset_SecondPhase(Dataset):
    """
    Dataset class for UBC Fashion dataset for second phase of training.
    Look at README.md for information to download and add poses to it.
    """
    def __init__(self, path, distr="train", inference=False):
        """
        Constructor for UBC Fashion dataset class.

        Args:
            path (str): path to UBC Fashion dataset dir containing all the scenes.
            distr (str): which distribution of dataset to load (train, test)
        """
        assert distr in ["train", "test"]

        self.base_path = os.path.join(path, distr, "scenes")
        self.distr = distr
        self.ids = [scene_id for scene_id in os.listdir(self.base_path) if scene_id.isdigit()]

        self.resolution = 512
        self.inference = inference

    def __len__(self):
        """
        Returns number of scenes in this dataset distribution.
        """
        return len(self.ids)
    
    def __getitem__(self, index):
        """
        Second phase of training is on 24-frame videos (probably 1 second segments).
        This method will return a randomly chosen contiguous 24-frames of a scene.

        Args:
            index (int): index of scene to get.

        Returns:
            ref_images (tensor): rgb image .shape=(1, self.resolution, self.resolution, 3)
            expected_images (tensor): rgb image .shape=(24, self.resolution, self.resolution, 3)
            expected_poses (tensor): pose image of the rgb image .shape=(24, self.resolution, self.resolution, 3)
        """
        time_frames = 4  # 24 frames uses too much vram
        scene_id = self.ids[index]
        scene_path = os.path.join(self.base_path, scene_id)
        image_filenames = sorted(glob.glob(os.path.join(scene_path, "images", "*.png")))
        if self.inference:
            time_frames = min(100, len(image_filenames))

        # randomly choose one image as reference
        ref_filename = random.choice(image_filenames)

        # if there are less than 24 frames, pad it with the last frame
        while len(image_filenames) < 4:
            image_filenames.append(image_filenames[-1])

        # randomly choose a contiguous 24 frames
        start_idx = random.randint(0, len(image_filenames) - time_frames)
        image_filenames = image_filenames[start_idx:start_idx+time_frames]
        pose_filenames = [os.path.join(scene_path, "dwposes", os.path.basename(f)) for f in image_filenames]

        # load images
        def load_img(path, fill_color=(255, 255, 255)):
            img = imageio.imread(path)
            # first three channels
            img = img[..., :3]
            # center crop to self.resolution x self.resolution
            img = center_fill(img, self.resolution, color=fill_color)
            # normalize
            img = img / 255.0
            img = 2.0 * img - 1.0
            # convert to tensor
            img = torch.from_numpy(img)
            return img

        images = torch.stack([load_img(f) for f in image_filenames])
        poses = torch.stack([load_img(f, fill_color=(0, 0, 0)) for f in pose_filenames])

        return {
            "ref_images": load_img(ref_filename).unsqueeze(0),
            "expected_images": images,
            "expected_poses": poses,
        }
