"""Script for demonstrating NeRF (Neural Radiance Fields) with a few features
removed.

This script is based off of tiny_nerf.ipynb from https://github.com/bmild/nerf
Just like the notebook, this version of NeRF will also use 3D input instead of
the 5D input from the complete version of NeRF. Colors and textures will look
about the same from different camera viewing angles. There is also no
hierarchical sampling so scenes will take longer to render.

Example usage:
Download tiny_nerf_data.npz from http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
python3 main.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

L_EMBED = 6  # Number of fourier features to encode xyz with.

def main():
    images, poses, focal_len, test_img, test_pose = load_data()
    model = MLP()

def load_data():
    """Load data for training."""
    data = np.load("tiny_nerf_data.npz")
    images = data["images"]  # A bunch of 100x100 RGB images of target object.
    poses = data["poses"]  # A bunch of 4x4 camera-to-world transformation matrices.
    focal_len = data["focal"]
    test_img, test_pose = images[101], poses[101]
    images, poses = images[:100], poses[:100]
    return images, poses, focal_len, test_img, test_pose

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 + 3*2*L_EMBED, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256 + self.fc1.in_features, 256)  # For skip connection.
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 4)

    def forward(self, x):
        orig_input = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x, orig_input)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        x = self.fc9(x)
        return x

if __name__ == "__main__":
    main()

