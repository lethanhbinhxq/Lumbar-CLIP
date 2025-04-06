import os
import numpy as np
from tiger.io import read_image
import matplotlib.pyplot as plt

# Directories (relative paths based on the script's location in Spider folder)
script_dir = os.path.dirname(os.path.abspath(__file__))  
input_dir = os.path.join(script_dir, "raw_data", "images")  
output_dir = os.path.join(script_dir, "preprocessed_images") 

# Read normalize_z_axis.txt
normalize_z_file_path = os.path.join(script_dir, "normalize_z_axis.txt")
with open(normalize_z_file_path, 'r') as f:
    normalize_z_names = set(line.strip() for line in f if line.strip())

os.makedirs(output_dir, exist_ok=True)

mha_files = [f for f in os.listdir(input_dir) if f.endswith(".mha")]

for mha_file in mha_files:
    input_path = os.path.join(input_dir, mha_file)
    
    try:
        image, header = read_image(input_path)

        mha_name = os.path.splitext(mha_file)[0]
    
        if mha_name in normalize_z_names:
            z = image.shape[2] // 2
            slice_img = image[:, :, z]
            slice_img_rotated = np.rot90(slice_img, k=-1)
            slice_img_transformed = np.fliplr(slice_img_rotated)

        else:
            x = image.shape[0] // 2
            slice_img = image[x, :, :]
            slice_img_transformed = np.rot90(slice_img)

        output_path = os.path.join(output_dir, f"{os.path.splitext(mha_file)[0]}.png")
    
        plt.imsave(output_path, slice_img_transformed, cmap="gray")

        print(f"Processed: {mha_file}")
    
    except Exception as e:
        print(f"Failed to process {mha_file}: {e}")

print(f"Conversion complete. PNG files saved to {output_dir}")
