import os
import xml.etree.ElementTree as ET
from numpy import zeros, asarray
import mrcnn.utils
import mrcnn.config
import mrcnn.model

class FSLDataset(mrcnn.utils.Dataset):
    
    def load_dataset(self, images_dir, annotations_dir, is_train=True):
        self.add_class("dataset", 1, "FSL")
        
        # Loop through the image files in the 'images/' directory
        for filename in os.listdir(images_dir):
            # Strip the file extension to get the image ID
            image_id, extension = os.path.splitext(filename)
            
            # Continue only if the file is an image (adjust the condition to match your image file types)
            if extension.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            # Determine whether the image should be part of the training or validation set
            if is_train and int(image_id) >= 40:
                continue
            if not is_train and int(image_id) < 40:
                continue

            # Construct the full paths to the image file and its corresponding annotation file
            img_path = os.path.join(images_dir, filename)
            ann_path = os.path.join(annotations_dir, image_id + '.xml')

            # Add the image to the dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
            
    # Loads the binary masks for an image.
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('FSL'))
        return masks, asarray(class_ids, dtype='int32')
    
# A helper method to extract the bounding boxes from the annotation file
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
    NUM_CLASSES = 2  # Background + FSL
    STEPS_PER_EPOCH = 6

def train_model_for_subfolder(subfolder_name, dataset_dir='FSL', epochs=1):
    # Define full paths for images and annotations directories within the subfolder
    images_dir = os.path.join(dataset_dir, subfolder_name, 'images/')
    annotations_dir = os.path.join(dataset_dir, subfolder_name, 'annotations/')

    # Initialize and prepare the datasets
    train_dataset = FSLDataset()
    train_dataset.load_dataset(images_dir=images_dir, annotations_dir=annotations_dir, is_train=True)
    train_dataset.prepare()

    validation_dataset = FSLDataset()
    validation_dataset.load_dataset(images_dir=images_dir, annotations_dir=annotations_dir, is_train=False)
    validation_dataset.prepare()

    # Model Configuration and Initialization
    fsl_config = FSLConfig()
    model = mrcnn.model.MaskRCNN(mode='training', model_dir='./', config=fsl_config)
    model.load_weights(filepath='mask_rcnn_gesture_0001.h5', by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    # Training
    model.train(train_dataset=train_dataset, val_dataset=validation_dataset, 
                learning_rate=fsl_config.LEARNING_RATE, epochs=epochs, layers='heads')

    # Save the model with a name specific to the subfolder
    model_path = f'model/{subfolder_name}_model.h5'
    model.keras_model.save_weights(model_path)

# List of subfolders to iterate through
subfolders = ['Fast', 'Five']

# subfolders = ['Fast', 'Five', 'Four', 'Hello', 'Imfine', 'One', 'Three', 'Two', 'Wrong', 'Yes']

# Loop through each subfolder and train the model
for subfolder in subfolders:
    print(f"Training model for {subfolder} subfolder.")
    train_model_for_subfolder(subfolder_name=subfolder, dataset_dir='FSL', epochs=1)
    print(f"Model trained and saved for {subfolder} subfolder.")

def extract_features(model_path, dataset, dataset_dir, subfolder_name):
    """Extract features from images using the trained Mask R-CNN model."""
    config = FSLConfig()
    model = mrcnn.model.MaskRCNN(mode="inference", config=config, model_dir=os.path.join('model', subfolder_name))
    model.load_weights(model_path, by_name=True)
    
    images_dir = os.path.join(dataset_dir, subfolder_name, 'images/')
    features = []  # List to hold features
    
    # Loop through images
    for image_id in dataset.image_ids:
        image_info = dataset.image_info[image_id]
        image_path = image_info['path']
        image = dataset.load_image(image_id)
        results = model.detect([image], verbose=0)
        r = results[0]
        
        # Example: Extract bounding box coordinates
        # Modify this part to extract the features you need
        bboxes = r['rois']  # [N, (y1, x1, y2, x2)]
        features.append(bboxes)
    
    # Convert to numpy array and save to .npy file
    features = np.array(features, dtype=object)  # Use dtype=object for variable-length sequences
    np.save(f'features_{subfolder_name}.npy', features)

for subfolder in subfolders:
    print(f"Training model for {subfolder} subfolder.")
    model_path = train_model_for_subfolder(subfolder_name=subfolder, dataset_dir='FSL', epochs=1)
    print(f"Model trained and saved for {subfolder} subfolder.")

    # Load the dataset again for feature extraction
    dataset = FSLDataset()
    dataset.load_dataset(images_dir=os.path.join('FSL', subfolder, 'images/'),
                         annotations_dir=os.path.join('FSL', subfolder, 'annots/'), is_train=False)
    dataset.prepare()

    # Extract and save features
    extract_features(model_path, dataset, 'FSL', subfolder)
