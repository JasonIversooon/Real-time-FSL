
# Real-time-FSL

Real-time-FSL is a system for real-time Filipino Sign Language (FSL) recognition using deep learning, computer vision, and a user-friendly GUI. It combines Mask R-CNN for hand segmentation and LSTM for gesture sequence classification.

## Features
- **Real-time gesture recognition** using webcam input
- **Mask R-CNN** for hand segmentation and feature extraction
- **LSTM** for temporal sequence modeling of gestures
- **Tkinter GUI** for live demo and user interaction
- **Data preprocessing** scripts for annotation, normalization, and keypoint extraction

## Project Structure

- `GUI/`
  - `app.py`: Main Tkinter GUI application. Captures webcam, runs detection, and displays predictions.
  - `model.py`: Loads the trained LSTM model and weights.
  - `utils.py`: Utility functions for MediaPipe detection, drawing, and keypoint extraction.
  - `test.py`: Minimal Flask app for testing web server setup.
  - `Real-Time_Inference.ipynb`: Jupyter notebook for real-time inference and experimentation.
  - `templates/index.html`: Web template for Flask-based demo (not used by Tkinter GUI).
  - `image.png`, `logo.jpg`: Images for GUI or documentation.

- `LSTM/`
  - `Real-Time_Inference.ipynb`: Notebook for LSTM-based gesture recognition, training, and evaluation.
  - `Evaluation.ipynb`, `MRCNN__LSTM_dataset.ipynb`: Notebooks for dataset creation and model evaluation.

- `mask r-cnn/`
  - `FSL.py`: Main Mask R-CNN training and dataset management script. Defines `FSLDataset` and `FSLConfig` classes, and `train_model()` for training on FSL data.
  - `fsltrial.py`: Alternative Mask R-CNN training and feature extraction pipeline, including per-class training and feature saving.
  - `keypoints_creation.py`: Extracts hand keypoints from segmented images using MediaPipe.
  - `normalization.py`: Normalizes images for consistent input to models.
  - `renaming.py`: Renames image and annotation files for dataset consistency.

## Getting Started

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare models**
   - Place your trained LSTM model and weights in `GUI/model/` as `Finale_modelo4.h5` and `Finale_weights4.h5` (or update `model.py` accordingly).
   - For Mask R-CNN, see `mask r-cnn/FSL.py` and `mask r-cnn/fsltrial.py` for training instructions and dataset structure.
4. **Run the GUI**
   ```bash
   cd GUI
   python app.py
   ```
5. **(Optional) Run Flask test server**
   ```bash
   python test.py
   ```

## Main Scripts and Notebooks

- **GUI/app.py**: Launches the Tkinter GUI, captures webcam, runs detection, and displays results.
- **GUI/model.py**: Loads the LSTM model for gesture classification.
- **GUI/utils.py**: Contains MediaPipe detection, drawing, and keypoint extraction utilities.
- **mask r-cnn/FSL.py**: Trains Mask R-CNN on FSL dataset, manages data splits, and saves weights.
- **mask r-cnn/fsltrial.py**: Trains Mask R-CNN per class, extracts features, and saves them for LSTM.
- **mask r-cnn/keypoints_creation.py**: Extracts hand keypoints from images using MediaPipe.
- **mask r-cnn/normalization.py**: Normalizes images for model input.
- **mask r-cnn/renaming.py**: Renames files for dataset consistency.
- **LSTM/Real-Time_Inference.ipynb**: End-to-end pipeline for LSTM-based gesture recognition.

## Requirements
See `requirements.txt` for a list of dependencies. Main libraries include:
- flask, numpy, opencv-python, pillow, torch, torchvision, matplotlib, scikit-learn, tensorflow, mediapipe, mrcnn

## Dataset Structure
The FSL dataset should be organized as follows:
```
FSL/
  ├── <ClassName>/
  │     ├── images/
  │     └── annotations/
```
Each class folder contains images and annotation XMLs for Mask R-CNN.

## License
MIT License
