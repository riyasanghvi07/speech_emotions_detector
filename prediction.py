# prediction.py

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

class EmotionPredictor:
    def __init__(self, model_path='./models/final_emotion_recognition_model.h5'):
        self.model = load_model(model_path)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

    @staticmethod
    def extract_mfcc(file_path):
        """Extract MFCC features from an audio file."""
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc

    def preprocess_audio(self, file_path):
        """Preprocess the audio file for prediction."""
        mfcc = self.extract_mfcc(file_path)
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
        mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension
        return mfcc

    def predict_emotion(self, file_path):
        """Predict the emotion of the given audio file."""
        processed_audio = self.preprocess_audio(file_path)
        prediction = self.model.predict(processed_audio)
        predicted_index = np.argmax(prediction[0])
        predicted_emotion = self.emotions[predicted_index]
        confidence = prediction[0][predicted_index]
        return predicted_emotion, confidence

def main():
    # Initialize the predictor
    predictor = EmotionPredictor()

    while True:
        # Get audio file path from user
        audio_file = input("Enter the path to the audio file (or 'q' to quit): ").strip()
        
        if audio_file.lower() == 'q':
            print("Exiting the program.")
            break

        if not os.path.exists(audio_file):
            print(f"Error: The file '{audio_file}' does not exist. Please enter a valid file path.")
            continue

        try:
            emotion, confidence = predictor.predict_emotion(audio_file)
            print(f"\nPredicted emotion: {emotion}")
            print(f"Confidence: {confidence:.2f}")
            print()  # Add a blank line for better readability
        except Exception as e:
            print(f"An error occurred while processing the file: {str(e)}")
            print("Please make sure the file is a valid audio file and try again.")
            print()  # Add a blank line for better readability

if __name__ == "__main__":
    main()