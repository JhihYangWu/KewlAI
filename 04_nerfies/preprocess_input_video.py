# based off of https://colab.research.google.com/github/google/nerfies/blob/main/notebooks/Nerfies_Capture_Processing.ipynb

import cv2
import os
from pathlib import Path
import random

# feel free to change
INPUT_VIDEO_PATH = "input_video/jhihyang.mov"  # path to input video
NERFIE_OUT_PATH = Path("jhihyang_nerfie/")  # path to save the final nerfie
SCALE = 0.3  # for decreasing nerfie quality for performance (1.0 = original, 0.1 = 1/10 quality)
VERT_FLIP = True  # whether to flip video vertically

# probably don't change these
TARGET_NUM_IMGS = 100
MIN_NUM_MATCHES = 32
FILTER_MAX_REPROJ_ERROR = 2
TRI_COMPLETE_MAX_REPROJ_ERROR = 2

def main():
    create_out_folders()
    cvt_vid_to_imgs()
    extract_features()
    match_features()
    reconstruction()

def create_out_folders():
    print("Creating output folders...")
    if os.path.exists(NERFIE_OUT_PATH): raise RuntimeError(f"Please delete {NERFIE_OUT_PATH}")
    os.mkdir(NERFIE_OUT_PATH)
    os.mkdir(NERFIE_OUT_PATH / "tmp/")
    os.mkdir(NERFIE_OUT_PATH / "tmp/images/")

def cvt_vid_to_imgs():
    print("Sampling images from video...")
    # extracts a few dozen images from the input video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    cap_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if cap_num_frames < TARGET_NUM_IMGS:
        raise RuntimeError("The input video was too short.")
    sample_freq = TARGET_NUM_IMGS / cap_num_frames
    j = 0
    for i in range(cap_num_frames):
        ret, frame = cap.read()
        if not ret: assert False
        if random.random() > sample_freq: continue  # so we write approx TARGET_NUM_IMGS
        if VERT_FLIP: frame = cv2.flip(frame, 0)
        frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
        cv2.imwrite(str(NERFIE_OUT_PATH / f"tmp/images/{j}.png"), frame)
        j += 1

def extract_features():
    print(f"Extracting sift features...")
    # extract sift features for SfM
    exit_code = os.system(f"colmap feature_extractor "
                          f"--SiftExtraction.use_gpu 0 "
                          f"--SiftExtraction.upright 1 "
                          f"--ImageReader.camera_model OPENCV "
                          f"--ImageReader.single_camera 1 "
                          f"--database_path {NERFIE_OUT_PATH / 'tmp/colmap_db'} "
                          f"--image_path {NERFIE_OUT_PATH / 'tmp/images/'} "
                          f"> /dev/null 2>&1")
    assert exit_code == 0

def match_features():
    print("Matching those features between images...")
    # match sift features between images
    exit_code = os.system(f"colmap exhaustive_matcher "
                          f"--SiftMatching.use_gpu 0 "
                          f"--database_path {NERFIE_OUT_PATH / 'tmp/colmap_db'} "
                          f"> /dev/null 2>&1")
    assert exit_code == 0

def reconstruction():
    print(f"Running SfM...")
    # run structure-from-motion
    exit_code = os.system(f"colmap mapper "
                          f"--Mapper.ba_refine_principal_point 1 "
                          f"--Mapper.filter_max_reproj_error {FILTER_MAX_REPROJ_ERROR} "
                          f"--Mapper.tri_complete_max_reproj_error {TRI_COMPLETE_MAX_REPROJ_ERROR} "
                          f"--Mapper.min_num_matches {MIN_NUM_MATCHES} "
                          f"--database_path {NERFIE_OUT_PATH / 'tmp/colmap_db'} "
                          f"--image_path {NERFIE_OUT_PATH / 'tmp/images/'} "
                          f"--output_path {NERFIE_OUT_PATH / 'tmp/'} "
                          f"> /dev/null 2>&1")
    assert exit_code == 0
    assert os.path.exists(NERFIE_OUT_PATH / "tmp/0/cameras.bin")

if __name__ == "__main__":
    main()
