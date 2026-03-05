import os

import cv2
import numpy as np
from tqdm import tqdm


def generate_video(images: list | np.ndarray, output_path: str, fps: int = 24):
    print("Generating Video...")

    folder = os.path.dirname(output_path)
    os.makedirs(folder, exist_ok=True)

    width = images[0].shape[1]
    height = images[0].shape[0]

    cap_write = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for image in tqdm(images):
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cap_write.write(bgr_image)

    cap_write.release()

    print(f"Video saved in {output_path}")
