# written by JhihYang Wu <jhihyangwu@arizona.edu>
# useful utilities

import numpy as np
import cv2
import torch
import os

def center_crop(img, resolution):
    """
    Center crop an image to a square of resolution x resolution.

    Args:
        img (np.ndarray): image to crop.
        resolution (int): resolution to crop to.
    """
    # first center crop
    h, w = img.shape[:2]
    if h > w:
        start = (h - w) // 2
        img = img[start:start+w, :]
    elif w > h:
        start = (w - h) // 2
        img = img[:, start:start+h]

    # then resize to resolution
    img = cv2.resize(img, (resolution, resolution))

    return img

def center_fill(img, resolution, color=(255, 255, 255)):
    """
    Fill an image to a square of resolution x resolution.
    
    Args:
        img (np.ndarray): image to fill.
        resolution (int): resolution to fill to.
    """
    # empty image
    filled_img = np.ones((resolution, resolution, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)

    # resize larger dimension to resolution
    h, w = img.shape[:2]
    if h > w:
        img = cv2.resize(img, (int(w / h * resolution), resolution))
    elif w > h:
        img = cv2.resize(img, (resolution, int(h / w * resolution)))
    else:
        img = cv2.resize(img, (resolution, resolution))
    
    # replace center of filled image with original image
    h, w = img.shape[:2]
    start_h = (resolution - h) // 2
    start_w = (resolution - w) // 2
    filled_img[start_h:start_h+h, start_w:start_w+w] = img

    return filled_img

#----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.
# Copied from https://github.com/NVlabs/edm/blob/main/torch_utils/misc.py

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

def img_tensor_to_npy(img_tensor):
    """
    Converts any range float image tensor to uint8 numpy array.

    Args:
        img_tensor (tensor): float tensor any range .shape=(H, W, C)
    
    Returns:
        img_npy (numpy arr): uint8 0 255 numpy array .shape=(H, W, C)
    """
    img_tensor = img_tensor.to(torch.float32)
    img_tensor = img_tensor - img_tensor.min()
    img_tensor = img_tensor / img_tensor.max()
    img_npy = img_tensor * 255
    img_npy = img_npy.cpu().detach().numpy().astype(np.uint8)
    return img_npy
