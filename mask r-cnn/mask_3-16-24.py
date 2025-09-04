import os
import cv2
import numpy as np
from tqdm import tqdm
import mrcnn.config
import mrcnn.model
from mrcnn import utils

# Adjust Mask R-CNN configuration if necessary
class InferenceConfig(mrcnn.config.Config):
    NAME = "inference_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  # Background + hand classes
    BACKBONE = 'resnet101'

def process_image(image_path, output_image_path, model):
    # Load the image
    image = cv2.imread(image_path)
    if image is not None:
        # Convert the image from BGR to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform prediction with Mask R-CNN
        results = model.detect([image_rgb], verbose=0)
        r = results[0]
        
        # Create a mask placeholder
        mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        
        # Accumulate masks of the detected objects (for simplicity, we assume every detected object is a hand)
        for i in range(r['masks'].shape[2]):
            mask[r['masks'][:,:,i]] = 1
        
        # Color the masked regions on the image - you can adjust the color
        colored_mask = np.zeros_like(image_rgb)
        colored_mask[mask == 1] = [0, 255, 0] # Green color for the mask
        
        # Blend the original image with the colored mask
        blended_image = cv2.addWeighted(image_rgb, 0.5, colored_mask, 0.5, 0)
        
        # Convert the blended image back to BGR format for OpenCV compatibility
        output_image = cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR)
        
        # Save the output image
        cv2.imwrite(output_image_path, output_image)
        print(f"Processed and saved: {output_image_path}")

if __name__ == "__main__":
    config = InferenceConfig()
    model = mrcnn.model.MaskRCNN(mode="inference", config=config, model_dir='./model_dir')
    model.load_weights('model/mask_rcnn_coco.h5', by_name=True)
    
    frames_root_dir = 'extracted_frames_v3/'
    output_dir = 'masked_images/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label_dir in os.listdir(frames_root_dir):
        label_path = os.path.join(frames_root_dir, label_dir)
        label_output_path = os.path.join(output_dir, label_dir)
        if not os.path.exists(label_output_path):
            os.makedirs(label_output_path)
        print(f"Processing label directory: {label_dir}")
        for image_folder in os.listdir(label_path):
            image_folder_path = os.path.join(label_path, image_folder)
            image_output_path = os.path.join(label_output_path, image_folder)
            if not os.path.exists(image_output_path):
                os.makedirs(image_output_path)
            print(f"  Processing image folder: {image_folder}")
            for image_file in sorted(os.listdir(image_folder_path)):
                image_path = os.path.join(image_folder_path, image_file)
                output_image_path = os.path.join(image_output_path, os.path.splitext(image_file)[0] + ".jpg")
                process_image(image_path, output_image_path, model)
            print(f"  Completed processing image folder: {image_folder}")
        print(f"Completed processing label directory: {label_dir}")