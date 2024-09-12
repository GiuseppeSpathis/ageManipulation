import argparse
from pathlib import Path
import multiprocessing as mp
import time
import os
import gc
import torch
from PIL import Image
from face_alignment import align
import numpy as np
from typing import List
import logging

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from inference import load_pretrained_model, to_input  # Assumendo che to_input sia definita in inference

# Caricare il modello pre-addestrato
model_path = './pretrained/adaface_ir50_webface4m.ckpt'
model = load_pretrained_model('ir_50')
model.eval()

def get_all_images(path: Path) -> List[str]:
    return sorted(list(path.rglob("*.png")) + list(path.rglob("*.jpg")))

def setup_logger():
    logger = logging.getLogger('EmbeddingLogger')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger()

def compute_embedding(image_path: Path, model) -> None:
    attempts = 0
    if image_path.with_suffix(".npy").exists():
        return
    while True:
        try:
            logger.info(f"provo ad allineare {image_path} al tentativo {attempts}")
            aligned_rgb_img = align.get_aligned_face(str(image_path))
            if aligned_rgb_img is None:
                raise ValueError(f"Face alignment failed for {image_path}")
            bgr_input = to_input(aligned_rgb_img)
            break
        except Exception as e:
            wait = 100 * (2 ** attempts)
            logger.error(f"Failed to load and align {image_path}, retrying in {wait} milliseconds due to {e}")
            time.sleep(wait / 1000)
            attempts += 1
            if attempts >= 10:
                raise
    with torch.no_grad():
        feature, _ = model(bgr_input)

    

    attempts = 0
    while True:
        try:
            with open(image_path.with_suffix(".npy"), "wb") as f:
                np.save(f, feature.cpu().numpy())
            logger.info(f"Saved embedding for {image_path} to {image_path.with_suffix('.npy')}")
            break
        except Exception as e:
            wait = 100 * (2 ** attempts)
            logger.error(f"Failed to save {image_path.with_suffix('.npy')}, retrying in {wait} milliseconds due to {e}")
            time.sleep(wait / 1000)
            attempts += 1
            if attempts >= 10:
                raise
    del feature
    gc.collect()

def process_image(image_path: Path) -> None:
    compute_embedding(image_path, model)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if args.files:
        images = args.files
    elif args.dataset:
        images = get_all_images(args.dataset)
    else:
        raise ValueError("Either files or --dataset must be specified")

    if args.num_workers == 0:
        for image in images:
            process_image(image)
    else:
        mp.set_start_method('spawn')  # Use 'spawn' start method
        with mp.Pool(args.num_workers) as pool:
            pool.map(process_image, images)

if __name__ == "__main__":
    main()

