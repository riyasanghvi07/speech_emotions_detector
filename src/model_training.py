# model_training.py

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

class ModelTraining:
    def __init__(self, model, X, y, checkpoint_filepath='./models/best_model.keras'):
        self.model = model
        self.X = X
        self.y = y
        self.checkpoint_filepath = checkpoint_filepath
        self.history = None

    def create_callbacks(self):
        model_checkpoint_callback = ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )

        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        lr_scheduler_callback = LearningRateScheduler(scheduler)

        return [model_checkpoint_callback, lr_scheduler_callback]

    def train_model(self, validation_split=0.2, epochs=10, batch_size=64):
        callbacks = self.create_callbacks()
        self.history = self.model.fit(
            self.X, 
            self.y, 
            validation_split=validation_split, 
            epochs=epochs, 
            batch_size=batch_size, 
            callbacks=callbacks
        )

    def get_history(self):
        return self.history

    def save_model(self, filepath='./models/complete_model.h5'):
        """
        Save the complete model after training.
        
        :param filepath: Path where the model will be saved
        """
        self.model.save(filepath)
        print(f"Complete model saved to {filepath}")

def main():
    # This is just for demonstration purposes
    # You would typically import your model and data here
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    import numpy as np

    # Create a dummy model and data for demonstration
    model = Sequential([Dense(1, input_shape=(1,))])
    model.compile(optimizer='adam', loss='mse')
    X = np.array([[1], [2], [3], [4]])
    y = np.array([[2], [4], [6], [8]])

    trainer = ModelTraining(model, X, y)
    trainer.train_model(epochs=20)  # Reduced epochs for demonstration
    history = trainer.get_history()
    print("Training complete. Access the history with trainer.get_history()")
    
    # Save the complete model
    trainer.save_model()

if __name__ == "__main__":
    main()