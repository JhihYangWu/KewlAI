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
from matplotlib.widgets import Slider
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Torch device: {device}")

L_EMBED = 6  # Number of fourier features to encode xyz with.
N_SAMPLES = 64  # Number of distances to sample per ray.
N_ITERS = 1000
BATCH_SIZE = 1024 * 32  # Number of points to feed into model at a single time.
LEARNING_RATE = 5e-4

def main():
    images, poses, focal_len, test_img, test_pose = load_data()
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if not os.path.exists("model.pt"):
        debug_loss(model, images[0], poses[0], focal_len, optimizer)  # Make sure loss is decreasing.
        for i in range(N_ITERS):
            rand_i = np.random.randint(images.shape[0])
            improve_model(model, images[rand_i], poses[rand_i], focal_len, optimizer)
            # Show image rendered from test_pose every 25 iterations.
            if i % 25 == 0:
                evaluate(model, test_img, test_pose, focal_len, i)
                print("Saving model.")
                save_model(model)
        plt.show()
    else:
        print("Found trained model. If you would like to train again, delete or rename model.pt.")
        model = load_model()
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25)
        axtheta = fig.add_axes([0.1, 0.05, 0.8, 0.05])
        axphi = fig.add_axes([0.1, 0.10, 0.8, 0.05])
        axradius = fig.add_axes([0.1, 0.15, 0.8, 0.05])
        theta_slider = Slider(
            ax=axtheta,
            label="Theta",
            valmin=0,
            valmax=360,
            valinit=100
        )
        phi_slider = Slider(
            ax=axphi,
            label="Phi",
            valmin=-90,
            valmax=0,
            valinit=-30
        )
        radius_slider = Slider(
            ax=axradius,
            label="Radius",
            valmin=3,
            valmax=5,
            valinit=4
        )
        def update_render(val):
            theta = theta_slider.val
            phi = phi_slider.val
            radius = radius_slider.val
            pred_img = render_interactive(model, focal_len, theta, phi, radius)
            ax.imshow(pred_img)
        theta_slider.on_changed(update_render)
        phi_slider.on_changed(update_render)
        radius_slider.on_changed(update_render)
        update_render(-1)
        plt.show()

def load_model():
    """Load pretrained model."""
    model = MLP()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    return model

def render_interactive(model, focal_len, theta, phi, radius):
    """Render a image using model interactively."""
    pose = create_pose_matrix(theta, phi, radius)
    H, W = 100, 100
    world_coord_d, world_coord_o = get_rays(H, W, focal_len, pose)
    pred_img = render_img(model, world_coord_d, world_coord_o,
                          z_near=2, z_far=6)
    pred_img = pred_img.cpu().detach().numpy()
    return pred_img

def create_pose_matrix(theta, phi, radius):
    """Create pose matrix given parameters."""
    # I have no idea of the math behind this. I just copied from tiny_nerf.ipynb.
    def trans_t(t):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1]
        ])
    def rot_phi(phi):
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1]
        ])  # Looks like 2D rotation matrix!
    def rot_theta(th):
        return np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1]
        ])  # Looks like 2D rotation matrix!
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180*np.pi) @ c2w
    c2w = rot_theta(theta/180*np.pi) @ c2w
    c2w = np.array([
        [-1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1]]) @ c2w
    return torch.tensor(c2w, dtype=torch.float32)

def debug_loss(model, img, pose, focal_len, optimizer):
    """Make sure loss can decrease by training for 3 iterations."""
    losses = np.zeros(3)
    for i in range(3):
        loss = improve_model(model, img, pose, focal_len, optimizer)
        losses[i] = loss
    not_improving = np.all(losses == losses[0])
    if not_improving:
        print("Loss is not decreasing. Please rerun script.")
        import sys
        sys.exit()
    else:
        print("Loss decreasing. Starting to use random images.")

def evaluate(model, true_img, test_pose, focal_len, iter):
    """See how well model performs on a unseen pose."""
    H, W = true_img.shape[:2]
    world_coord_d, world_coord_o = get_rays(H, W, focal_len, test_pose)
    pred_img = render_img(model, world_coord_d, world_coord_o,
                          z_near=2, z_far=6)
    pred_img = pred_img.detach().numpy()
    loss = np.mean(np.square(pred_img - true_img))
    print("Testing loss:", loss)
    psnr = -10 * np.log(loss) / np.log(10)  # Peak signal-to-noise ratio.
    plt.title(f"ITER: {iter} | PSNR: {psnr}")
    plt.imshow(pred_img)
    plt.pause(0.1)

def save_model(model):
    """Save model weights."""
    torch.save(model.state_dict(), "model.pt")

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
        self.fc1 = nn.Linear(3 + 3*2*L_EMBED, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc3 = nn.Linear(256, 256, device=device)
        self.fc4 = nn.Linear(256, 256, device=device)
        self.fc5 = nn.Linear(256 + self.fc1.in_features, 256, device=device)  # For skip connection.
        self.fc6 = nn.Linear(256, 256, device=device)
        self.fc7 = nn.Linear(256, 256, device=device)
        self.fc8 = nn.Linear(256, 256, device=device)
        self.fc9 = nn.Linear(256, 4, device=device)

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
        x = self.fc5(torch.cat([x, orig_input], dim=-1))
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        x = self.fc9(x)
        return x

def improve_model(model, img, pose, focal_len, optimizer):
    """Improves the model a tiny bit."""
    # Use model to render image from pose.
    H, W = img.shape[:2]
    world_coord_d, world_coord_o = get_rays(H, W, focal_len, pose)
    pred_img = render_img(model, world_coord_d, world_coord_o, z_near=2, z_far=6)
    
    # Calculate loss.
    img_tensor = torch.tensor(img, dtype=torch.float32, requires_grad=False, device=device)
    loss = torch.mean(torch.square(pred_img - img_tensor))

    # Update model weights using gradients.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print stats.
    print("Training loss:", loss.item())

    return loss.item()

def get_rays(H, W, focal_len, pose):
    """
    Create partial world coordinates by partially converting image coordinates
    to world coordinates using camera-to-world matrix. Once these partial
    coordinates are multiplied by z (how far pinhole is from a point in
    space), they become world coordinates which can be fed into the model. 
    """
    xs = torch.arange(W, dtype=torch.float32)
    ys = torch.arange(H, dtype=torch.float32)
    i, j = torch.meshgrid(xs, ys, indexing="xy")
    i = (i - W/2) / focal_len
    j = -(j - H/2) / focal_len  # - sign because pinhole cameras flip image vertically.
    camera_coord = torch.stack([i, j, -torch.ones_like(i)], dim=-1)
    # tbh, I don't get why it is -ones_like.
    # camera_coord contains 100x100 of (x/z, y/z, -1).
    # Multiplying by z will give you x, y, -z in camera coord system.
    rot_matrix = pose[:3, :3]
    world_coord_d = torch.sum(camera_coord[..., None, :] * rot_matrix, dim=-1)
    offset_vector = torch.tensor(pose[:3, -1], dtype=torch.float32, device=device)
    world_coord_o = torch.broadcast_to(offset_vector, world_coord_d.shape)
    # world_coord_d and world_coord_o are separated for now because we need to
    # multiply world_coord_d with z independently later.
    return world_coord_d.to(device), world_coord_o

def render_img(model, world_coord_d, world_coord_o, z_near, z_far):
    """Render a image using model."""
    # Create 3D query points.
    z_vals = torch.linspace(z_near, z_far, N_SAMPLES, device=device)
    pts = (world_coord_d[..., None, :] * z_vals[:, None] +
           world_coord_o[..., None, :])

    # Run network.
    pts_flat = pts.reshape((-1, 3))
    pts_flat = add_fourier(pts_flat)
    raw = []
    for i in range(0, pts_flat.shape[0], BATCH_SIZE):
        raw.append(model(pts_flat[i:i+BATCH_SIZE]))
    raw = torch.cat(raw, dim=0)
    raw = raw.reshape(pts.shape[:-1] + (4,))
    
    # Compute opacities/transparencies and colors.
    rgb = torch.sigmoid(raw[..., :3])
    sigma = raw[..., 3].relu()

    # Do volume rendering.
    dists_btwn = torch.cat([z_vals[1:] - z_vals[:-1],
                            torch.tensor([1e10], dtype=torch.float32, device=device)])
    alpha = 1 - torch.exp(-sigma * dists_btwn)
    # alpha will be close to 1 if density/sigma is large and close to 0 otherwise.
    weights = alpha * torch.cumprod(1 - alpha + 1e-10, dim=-1)

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    _ = rgb_map.mean()  # Hack found to prevent loss from not decreasing.
    return rgb_map

def add_fourier(x):
    """Add fourier features to training set x."""
    retval = [x]
    for i in range(L_EMBED):
        for fn in [torch.sin, torch.cos]:
            retval.append(fn(2 * i * x))
    return torch.cat(retval, dim=-1)

if __name__ == "__main__":
    main()

