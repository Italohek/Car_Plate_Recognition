import cv2
import numpy as np
import os
from multiprocessing import Pool
from glob import glob


def main():
    # Path to your training images
    dataset_path = "dataset/train/images"
    output_path = "blurred/images"
    os.makedirs(output_path, exist_ok=True)

    images = glob(os.path.join(dataset_path, "*.jpg")) + \
            glob(os.path.join(dataset_path, "*.png"))

    for img_path in images:
        img = cv2.imread(img_path)
        blur = apply_motion_blur(img, size=7)

        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_path, filename), blur)
    
    print("Motion blur augmentation completed!")


def apply_motion_blur(image, size=15):
    # Create motion blur kernel
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size

    # Apply the kernel
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

if __name__ == "__main__":
    main()