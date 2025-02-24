import numpy as np
import rasterio
import os
from pathlib import Path
from scipy.ndimage import rotate

def load_and_normalize_image(filepath):
    with rasterio.open(filepath) as src:
        img = src.read()
        # Normalize globally across all bands instead of per band
        min_val = img.min()
        max_val = img.max()
        img = (img - min_val) / (max_val - min_val)
        return img

def generate_augmented_dataset(source_dir, output_dir, num_images=50, crop_size=64):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    source_files = [f for f in os.listdir(source_dir) if f.endswith('.tif')]
    source_images = [load_and_normalize_image(os.path.join(source_dir, f)) for f in source_files]
    
    count = 0
    while count < num_images:
        # Randomly select source image index
        img_idx = np.random.randint(0, len(source_images))
        img = source_images[img_idx]
        
        max_x = img.shape[1] - crop_size
        max_y = img.shape[2] - crop_size
        
        if max_x <= 0 or max_y <= 0:
            continue
            
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        cropped = img[:, x:x+crop_size, y:y+crop_size]
        
        # Random rotation
        angle = np.random.choice([0, 90, 180, 270])
        if angle != 0:
            cropped = rotate(cropped, angle, axes=(1, 2), reshape=False)
        
        # Random flip
        if np.random.random() > 0.5:
            cropped = np.flip(cropped, axis=1)
        if np.random.random() > 0.5:
            cropped = np.flip(cropped, axis=2)
            
        output_path = os.path.join(output_dir, f'AVIRIS_{count+1}.tif')
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=crop_size,
            width=crop_size,
            count=img.shape[0],
            dtype=np.float32
        ) as dst:
            dst.write(cropped)
            
        count += 1
        if count % 50 == 0:
            print(f"Generated {count} images")

if __name__ == "__main__":
    # source_dir = "/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project/HSI Data Sets/AVIRIS_Indian_Pine_SWIR_10_4231_R7RX991C/aviris_hyperspectral_data/"
    source_dir = "/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project/HSI Data Sets/AVIRIS_Indian_Pine_SWIR_10_4231_R7RX991C/T1"
    output_dir = "/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project/HSI Data Sets/AVIRIS_augmented_dataset_2/"
    generate_augmented_dataset(source_dir, output_dir)
