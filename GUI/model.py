import os
import sys
import tensorflow as tf

def load_model():
    # Path setup for PyInstaller
    base_path = getattr(sys, 'MEIPASS', os.path.dirname(os.path.abspath(_file)))
    model_path = os.path.join(base_path, 'model', 'Finale_modelo4.h5')
    weights_path = os.path.join(base_path, 'model', 'Finale_weights4.h5')

    # Load your TensorFlow model
    model = tf.keras.models.load_model(model_path)
    model.load_weights(weights_path)
    return model