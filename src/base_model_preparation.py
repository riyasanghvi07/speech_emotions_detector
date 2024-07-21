# base_model_preparation.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class BaseModelPreparation:
    def __init__(self, input_shape=(40, 1), num_classes=6):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        self.model = Sequential([
            LSTM(256, return_sequences=False, input_shape=self.input_shape),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])

    def compile_model(self, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        if self.model is None:
            raise ValueError("Model hasn't been built yet. Call build_model() first.")
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def get_summary(self):
        if self.model is None:
            raise ValueError("Model hasn't been built yet. Call build_model() first.")
        self.model.summary()

    def get_model(self):
        if self.model is None:
            raise ValueError("Model hasn't been built yet. Call build_model() first.")
        return self.model

def main():
    # This is just for demonstration purposes
    model_prep = BaseModelPreparation()
    model_prep.build_model()
    model_prep.compile_model()
    model_prep.get_summary()

    # If you need to use the model elsewhere
    model = model_prep.get_model()

if __name__ == "__main__":
    main()