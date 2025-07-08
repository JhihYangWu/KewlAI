from annotator.dwpose import DWposeDetector
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

TIKTOK_DATASET_PATH = "/workspace/jhihyangwu/data/tiktokdataset/TikTok_dataset/TikTok_dataset/"

def main():
    pose = DWposeDetector()
    scene_names = os.listdir(TIKTOK_DATASET_PATH)
    for i, scene in enumerate(scene_names):
        print(f"Scene {i+1}/{len(scene_names)}: {scene}")
        scene_path = os.path.join(TIKTOK_DATASET_PATH, scene)
        os.makedirs(os.path.join(scene_path, "dwposes"), exist_ok=True)
        image_names = os.listdir(os.path.join(scene_path, "images"))        
        for image in tqdm(image_names):
            image_path = os.path.join(scene_path, "images", image)
            oriImg = cv2.imread(image_path)
            out = pose(oriImg)
            out_path = os.path.join(scene_path, "dwposes", image)
            plt.imsave(out_path, out)

if __name__ == "__main__":
    main()
