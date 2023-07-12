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

def main():
    images, poses, focal_len, test_img, test_pose = load_data()

def load_data():
    """Load data for training."""
    data = np.load("tiny_nerf_data.npz")
    images = data["images"]  # A bunch of 100x100 RGB images of target object.
    poses = data["poses"]  # A bunch of 4x4 camera-to-world transformation matrices.
    focal_len = data["focal"]
    test_img, test_pose = images[101], poses[101]
    images, poses = images[:100], poses[:100]
    return images, poses, focal_len, test_img, test_pose

if __name__ == "__main__":
    main()

