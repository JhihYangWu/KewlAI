# written by JhihYang Wu <jhihyangwu@arizona.edu>

from torch.utils.data import Dataset
import random
import glob
import os
import imageio
import numpy as np
from animate_anyone.utils.utils import center_crop
import torch

TRAIN_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99, 100, 101, 104, 105, 106, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 152, 153, 154, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 183, 184, 185, 186, 187, 188, 189, 191, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 206, 207, 208, 209, 211, 212, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 230, 231, 232, 233, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 266, 267, 268, 269, 271, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 299, 300, 302, 303, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 339, 340]
TEST_IDS = [11, 12, 18, 19, 29, 38, 43, 46, 47, 57, 92, 102, 103, 109, 120, 124, 135, 151, 155, 163, 168, 182, 190, 192, 205, 210, 213, 227, 228, 229, 234, 254, 265, 270, 275, 298, 301, 304, 324, 338]

class TikTokDataset_FirstPhase(Dataset):
    """
    Dataset class for TikTok dataset [20] for first phase of training.
    Look at README.md for information to download and add poses to it.
    """
    def __init__(self, path, distr="train"):
        """
        Constructor for TikTok dataset class.

        Args:
            path (str): path to TikTok dataset dir containing all the scenes.
            distr (str): which distribution of dataset to load (train, test)
        """
        assert distr in ["train", "test"]

        self.base_path = path
        self.distr = distr
        self.ids = TRAIN_IDS if distr == "train" else TEST_IDS

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
        scene_id = f"{self.ids[index]:05}"
        scene_path = os.path.join(self.base_path, scene_id)
        image_filenames = sorted(glob.glob(os.path.join(scene_path, "images", "*.png")))
        # randomly choose one image as reference
        ref_filename = random.choice(image_filenames)
        # randomly choose another image as expected and corresponding pose image
        expected_filename = random.choice(image_filenames)
        expected_pose_filename = os.path.join(scene_path, "dwposes", os.path.basename(expected_filename))

        # load images
        def load_img(path):
            img = imageio.imread(path)
            # center crop to self.resolution x self.resolution
            img = center_crop(img, self.resolution)           
            # first three channels
            img = img[..., :3]
            # normalize
            img = img / 255.0
            img = 2.0 * img - 1.0  # so that it is in range [-1, 1]
            # convert to tensor
            img = torch.from_numpy(img)
            return img

        return {
            "ref_images": load_img(ref_filename).unsqueeze(0),
            "expected_images": load_img(expected_filename).unsqueeze(0),
            "expected_poses": load_img(expected_pose_filename).unsqueeze(0),
        }

class TikTokDataset_SecondPhase(Dataset):
    """
    Dataset class for TikTok dataset [20] for second phase of training.
    Look at README.md for information to download and add poses to it.
    """
    def __init__(self, path, distr="train"):
        """
        Constructor for TikTok dataset class.

        Args:
            path (str): path to TikTok dataset dir containing all the scenes.
            distr (str): which distribution of dataset to load (train, test)
        """
        assert distr in ["train", "test"]

        self.base_path = path
        self.distr = distr
        self.ids = TRAIN_IDS if distr == "train" else TEST_IDS

        self.resolution = 512

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
        time_frames = 4
        scene_id = f"{self.ids[index]:05}"
        scene_path = os.path.join(self.base_path, scene_id)
        image_filenames = sorted(glob.glob(os.path.join(scene_path, "images", "*.png")))

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
        def load_img(path):
            img = imageio.imread(path)
            # center crop to self.resolution x self.resolution
            img = center_crop(img, self.resolution)           
            # first three channels
            img = img[..., :3]
            # normalize
            img = img / 255.0
            img = 2.0 * img - 1.0
            # convert to tensor
            img = torch.from_numpy(img)
            return img

        images = torch.stack([load_img(f) for f in image_filenames])
        poses = torch.stack([load_img(f) for f in pose_filenames])

        return {
            "ref_images": load_img(ref_filename).unsqueeze(0),
            "expected_images": images,
            "expected_poses": poses,
        }
