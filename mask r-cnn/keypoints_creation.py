import os
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_image_with_mediapipe(image_path, npy_save_path):
    image = cv2.imread(image_path)
    if image is None:
        return
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use MediaPipe to detect hand keypoints
    mp_results = hands.process(image_rgb)
    keypoints = []

    if mp_results.multi_hand_landmarks:
        for hand_landmarks in mp_results.multi_hand_landmarks:
            # Extract keypoints for each hand
            hand_keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            keypoints.append(hand_keypoints)
    
    # Save keypoints if any hands were detected
    if keypoints:
        np.save(npy_save_path, np.array(keypoints))
        print(f"Keypoints saved to {npy_save_path}")

if __name__ == "__main__":
    processed_images_dir = 'masked_images/'  # Directory containing segmented images
    keypoints_dir = 'keypoints_sequence/'

    os.makedirs(keypoints_dir, exist_ok=True)
    
    for label_dir in os.listdir(processed_images_dir):
        label_path = os.path.join(processed_images_dir, label_dir)
        for image_folder in os.listdir(label_path):
            image_folder_path = os.path.join(label_path, image_folder)
            for image_file in os.listdir(image_folder_path):
                if image_file.endswith(".jpg"):
                    image_path = os.path.join(image_folder_path, image_file)
                    keypoints_output_path = os.path.join(keypoints_dir, label_dir, image_folder)
                    os.makedirs(keypoints_output_path, exist_ok=True)
                    npy_save_path = os.path.join(keypoints_output_path, os.path.splitext(image_file)[0] + ".npy")
                    process_image_with_mediapipe(image_path, npy_save_path)
