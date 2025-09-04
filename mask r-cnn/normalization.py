import cv2
import os
import numpy as np

# Define the path to the source root directory
source_root = 'FSL/'

# List of folder names to iterate through
folders = ['Fast', 'Five', 'Four', 'Hello', 'Imfine', 'One', 'Three', 'Two', 'Wrong', 'Yes']

# Function to normalize an image
def normalize_image(image):
    # Convert image to float32 for processing
    image = image.astype(np.float32)
    # Normalize image to range [0, 1]
    normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return normalized_image

# Iterate through each folder
for folder in folders:
    source_folder = os.path.join(source_root, folder, 'annotations', 'images')
    
    # Iterate through each image in the source directory
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct the full file path
            file_path = os.path.join(source_folder, filename)
            
            # Read the image
            image = cv2.imread(file_path)
            
            # Print original image pixel value range
            print(f"Original image pixel range for {filename}: {image.min()} to {image.max()}")
            
            # Normalize the image
            normalized_image = normalize_image(image)
            
            # Print normalized image pixel value range
            # Since normalized_image is in range [0, 1], multiply by 255 for equivalent 8-bit range
            print(f"Normalized image pixel range for {filename}: {normalized_image.min() * 255} to {normalized_image.max() * 255}")
            
            # Save the normalized image back to its original location. Since OpenCV can save only in uint8 format,
            # we need to convert the normalized image back to uint8.
            cv2.imwrite(file_path, (normalized_image * 255).astype(np.uint8))

print("Image normalization complete.")
