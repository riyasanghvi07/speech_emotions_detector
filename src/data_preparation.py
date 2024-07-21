import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import OneHotEncoder

class DataPreparation:
    def __init__(self, crema_path):
        self.crema_path = crema_path
        self.crema_df = None
        self.X = None
        self.y = None

    def create_dataframe(self):
        crema_dir_list = os.listdir(self.crema_path)
        emotions_crema = []
        paths_crema = []

        for it in crema_dir_list:
            paths_crema.append(os.path.join(self.crema_path, it))
            part = it.split('_')
            emotion = self.map_emotion(part[2])
            emotions_crema.append(emotion)

        emotions_crema_df = pd.DataFrame(emotions_crema, columns=['Emotions'])
        path_crema_df = pd.DataFrame(paths_crema, columns=['Path'])
        self.crema_df = pd.concat([emotions_crema_df, path_crema_df], axis=1)

    @staticmethod
    def map_emotion(emotion_code):
        emotion_map = {
            'SAD': 'sad',
            'ANG': 'angry',
            'DIS': 'disgust',
            'FEA': 'fear',
            'HAP': 'happy',
            'NEU': 'neutral'
        }
        return emotion_map.get(emotion_code, 'Unknown')

    @staticmethod
    def extract_mfcc(filename):
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc

    def process_audio_files(self):
        X_mfcc = self.crema_df['Path'].apply(lambda x: self.extract_mfcc(x))
        self.X = np.array([x for x in X_mfcc])
        self.X = np.expand_dims(self.X, -1)

    def encode_labels(self):
        enc = OneHotEncoder()
        self.y = enc.fit_transform(self.crema_df[['Emotions']])
        self.y = self.y.toarray()

    def prepare_data(self):
        self.create_dataframe()
        self.process_audio_files()
        self.encode_labels()

    def get_data(self):
        return self.X, self.y, self.crema_df

def main():
    crema_path = 'C:/Users/riyas/Desktop/EMOR/Crema/'
    data_prep = DataPreparation(crema_path)
    data_prep.prepare_data()
    X, y, crema_df = data_prep.get_data()

    print("Crema DataFrame shape:", crema_df.shape)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # You can add more processing or visualization here if needed

if __name__ == "__main__":
    main()