import os
import xml.etree.ElementTree as ET
from numpy import zeros, asarray
import mrcnn.utils
import mrcnn.config
import mrcnn.model
import numpy as np

class FSLDataset(mrcnn.utils.Dataset):
    
    def load_dataset(self, dataset_dir, class_names, is_train=True, train_val_split=0.8):
        # Add classes
        class_names = ['Fast', 'Five', 'Four', 'Hello', 'Imfine', 'One', 'Three', 'Two', 'Wrong', 'Yes']
        for class_id, class_name in enumerate(class_names, 1):  # Start IDs from 1
            self.add_class("dataset", class_id, class_name)
        
        # Iterate through all subfolders (one per class)
        for class_name in class_names:
            images_dir = os.path.join(dataset_dir, class_name, 'images/')
            annotations_dir = os.path.join(dataset_dir, class_name, 'annotations/')
            images = sorted(os.listdir(images_dir))  # Ensure consistent ordering
            
            # Determine split index for training and validation
            split_index = int(len(images) * train_val_split)
            
            for idx, filename in enumerate(images):
                image_id, extension = os.path.splitext(filename)
                if extension.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                # Apply split based on is_train flag and index
                if is_train and idx >= split_index:
                    continue
                if not is_train and idx < split_index:
                    continue
                
                img_path = os.path.join(images_dir, filename)
                ann_path = os.path.join(annotations_dir, image_id + '.xml')
                self.add_image('dataset', image_id=image_id + '_' + class_name, path=img_path, annotation=ann_path, class_id=class_names.index(class_name)+1)
                
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        
        class_ids = list()
        for i in range(len(boxes)):
            class_ids.append(info['class_id'])
        return masks, asarray(class_ids, dtype='int32')
    
    def extract_boxes(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()

        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)

        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

class FSLConfig(mrcnn.config.Config):
    NAME = "FSL_config"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 11  # Background + 10 classes
    STEPS_PER_EPOCH = 480

    # Use a custom image preprocessing function to apply data augmentation
    def image_preprocess(self, image, input_shape):
        # Apply data augmentation techniques
        image = mrcnn.utils.resize_image(image, min_dim=input_shape[0], max_dim=input_shape[1], mode='square')
        image, _, _, _ = mrcnn.utils.random_transform(image, input_shape, random_transform=False, augment=True)
        return image

def train_model(dataset_dir='FSL', epochs=5):
    class_names = ['Fast', 'Five', 'Four', 'Hello', 'Imfine', 'One', 'Three', 'Two', 'Wrong', 'Yes']
    
    # Initialize and prepare the dataset
    dataset = FSLDataset()
    dataset.load_dataset(dataset_dir=dataset_dir, class_names=class_names, is_train=True)
    dataset.prepare()

    validation_dataset = FSLDataset()
    validation_dataset.load_dataset(dataset_dir=dataset_dir, class_names=class_names, is_train=False)
    validation_dataset.prepare()
    
    # Model Configuration and Initialization
    config = FSLConfig()
    # Adjust STEPS_PER_EPOCH if necessary based on your dataset size and training preferences
    config.STEPS_PER_EPOCH = len(dataset.image_ids) // config.IMAGES_PER_GPU
    model = mrcnn.model.MaskRCNN(mode='training', model_dir='./', config=config)
    model.load_weights(filepath='mask_rcnn_gesture_0001.h5', by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    
    # Training
    model.train(train_dataset=dataset, val_dataset=validation_dataset, 
                learning_rate=config.LEARNING_RATE, epochs=epochs, layers='heads')
    
    # Save the model
    model_path = 'fsl_model_all_classes.h5'
    model.keras_model.save_weights(model_path)

    return model_path

# Train the model
model_path = train_model(dataset_dir='FSL', epochs=5)
print(f"Model trained and saved at {model_path}")