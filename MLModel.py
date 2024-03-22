import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import load_model
import pathlib
import numpy as np
from PIL import Image


#TODO Fixa så man kan välja lagert i nn och andra parametrar.

PATH_TO_MODEL = "model\model.keras"

class MLModel:
    def __init__(self, data_dir='./Training_directories', image_size=(180, 180), batch_size=32):
        self.data_dir = pathlib.Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.train_ds = None
        self.val_ds = None
        self.num_classes = None
        self.class_names = []

        for d in os.listdir("Training_directories"):
            self.class_names.append(f'Training_directories\{d}')
        
        
    def load_data(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=self.image_size,
            batch_size=self.batch_size
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=self.image_size,
            batch_size=self.batch_size
        )

        self.class_names = self.train_ds.class_names
        self.num_classes = len(self.class_names)

    def create_model(self):
        if not self.train_ds or not self.val_ds:
            raise ValueError("Data sets are not loaded. Call load_data() first.")

        self.model = Sequential([
            layers.Rescaling(1./255, input_shape=self.image_size + (3,)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes)
        ])

    def compile_and_train_model(self, epochs=10):
        if not self.model:
            raise ValueError("Model is not created. Call create_model() first.")

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.history = self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=epochs)

    def save_model(self, path=PATH_TO_MODEL):
        if self.model:
            self.model.save(path)
            print("Model saved at:", path)
        else:
            print("No model to save.")

    def get_training_history(self):
        if self.history:
            return self.history.history
        else:
            print("No training history available.")

    def train_and_save(self, epochs=10):
        self.load_data()
        self.create_model()
        self.compile_and_train_model(epochs)
        self.save_model()
        
    def predict(self, img):
        if not self.model:
            self.model = load_model(PATH_TO_MODEL)
        prediction = self.model.predict(self.format_image(img))
        
        predicted_class = np.argmax(prediction)
        self.prediction = self.class_names[predicted_class]
        
        self.confidence = self.calculate_confidence(prediction[0])
        
    def get_prediction(self,img):
        self.predict(img)
        return {"prediction": self.prediction, "confidence": self.confidence}
    
    
    def format_image(self, img):
        img = img.convert('RGB')
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = tf.image.resize(img, (180, 180))
        return img
    
    def calculate_confidence(self, logits):
        confidences = self.softmax(logits)
        confidence = np.max(confidences)
        
        confidence_as_precent = confidence * 100
        return round(confidence_as_precent, 2) 

    def softmax(self, logits):
        e = np.exp(logits - np.max(logits))
        return e / e.sum(axis=0)
