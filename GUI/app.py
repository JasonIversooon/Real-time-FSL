import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, prob_viz
from model import load_model

model = load_model()

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Define the colors for drawing
colors = [
    (245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16),
    (16,117,245), (245,66,230), (245,66,230), (121,44,250), (121,22,76)
]

actions = np.array(['Kape', 'Pinsan','Mabilis', 'Lima', 'Isa','Salamat', 'Tatlo','Dalawa','Mali','Oo'])

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition")
        self.sequence = []  # Initialize sequence here
        self.predictions = []  # Initialize predictions here
        self.threshold = 0.85  # Set threshold value
        self.sentence = []  # Initialize sentence here
        
        # Create a label to display the video feed
        self.video_label = tk.Label(root)
        self.video_label.pack()
        
        # Start video feed
        self.video_feed()
        
    def video_feed(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        with mp_holistic.Holistic(min_detection_confidence=0.65, min_tracking_confidence=0.65) as holistic:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Prediction logic
                keypoints = extract_keypoints(results)
                self.sequence.append(keypoints)  # Use self.sequence
                self.sequence = self.sequence[-60:]  # Keep the last 60

                if len(self.sequence) == 60:  # Use self.sequence
                    res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    self.predictions.append(np.argmax(res))  # Use self.predictions

                    # Visualization and logic for handling predictions
                    if np.unique(self.predictions[-30:])[0] == np.argmax(res):  # Adjusted to 10 for the example
                        if res[np.argmax(res)] > self.threshold:  # Use self.threshold
                            if len(self.sentence) > 0:
                                if actions[np.argmax(res)] != self.sentence[-1]:
                                    self.sentence.append(actions[np.argmax(res)])
                            else:
                                self.sentence.append(actions[np.argmax(res)])
                    if len(self.sentence) > 3:
                        self.sentence = self.sentence[-3:]  # Keep the last 5 actions

                    # Visualization on the frame
                    image = prob_viz(res, actions, image, colors)
                    cv2.rectangle(image, (0, 0), (image.shape[1], 40), (255, 105, 97), -1)  # Add background behind the text
                    cv2.putText(image, ' '.join(self.sentence), (3, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Convert image to RGB format for Tkinter
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                # Update video label with new frame
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                # Break the loop when the window is closed
                self.root.protocol("WM_DELETE_WINDOW", lambda: cap.release())
                
                self.root.update_idletasks()
                self.root.update()

        cap.release()

# Create Tkinter window
root = tk.Tk()
app = App(root)
root.mainloop()
