# main.py

from src.data_preparation import DataPreparation
from src.base_model_preparation import BaseModelPreparation
from src.model_training import ModelTraining

def main():
    # Data Preparation
    crema_path = 'C:/Users/riyas/Desktop/EMOR/Crema/'
    data_prep = DataPreparation(crema_path)
    data_prep.prepare_data()
    X, y, crema_df = data_prep.get_data()

    print("Crema DataFrame shape:", crema_df.shape)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Display the first few rows of the DataFrame
    print("\nFirst few rows of Crema DataFrame:")
    print(crema_df.head())

    # Display some statistics about the data
    print("\nData statistics:")
    print(crema_df['Emotions'].value_counts())

    # Model Preparation
    input_shape = (X.shape[1], 1)  # Assuming X is shaped as (n_samples, 40, 1)
    num_classes = y.shape[1]  # Assuming y is one-hot encoded

    model_prep = BaseModelPreparation(input_shape=input_shape, num_classes=num_classes)
    model_prep.build_model()
    model_prep.compile_model()
    model_prep.get_summary()

    # Get the model for training
    model = model_prep.get_model()

    # Model Training
    trainer = ModelTraining(model, X, y)
    trainer.train_model()

    # Get training history if needed
    history = trainer.get_history()

    print("\nTraining complete.")
    print("Final training accuracy:", history.history['accuracy'][-1])
    print("Final validation accuracy:", history.history['val_accuracy'][-1])

    # Save the complete model
    trainer.save_model('./models/final_emotion_recognition_model.h5')

    # You can add more processing, evaluation, or visualization here

if __name__ == "__main__":
    main()