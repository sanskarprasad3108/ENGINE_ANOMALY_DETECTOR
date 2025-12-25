import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


def main():
    df = pd.read_csv('dump_truck_engine_anomaly_dataset (1).csv')
    normal_data = df[df['label'] == 0]

    features = ['engine_temperature_C', 'oil_pressure_bar', 'engine_rpm', 'fuel_consumption_lph', 'vibration_mm_s']
    data = normal_data[features].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    model = Sequential([
        Dense(3, activation='relu', input_shape=(5,)),
        Dense(2, activation='relu'),
        Dense(3, activation='relu'),
        Dense(5, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='mae')

    print('Training Autoencoder...')
    model.fit(data_scaled, data_scaled, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

    reconstructions = model.predict(data_scaled)
    mae = np.mean(np.abs(data_scaled - reconstructions), axis=1)
    threshold = np.max(mae)
    print(f'Training Complete. Anomaly Threshold: {threshold}')

    os.makedirs('model_artifacts', exist_ok=True)
    model.save('model_artifacts/autoencoder_model.h5')
    with open('model_artifacts/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('model_artifacts/threshold.npy', 'wb') as f:
        np.save(f, threshold)

    print("Model, Scaler, and Threshold saved to 'model_artifacts/'")


if __name__ == '__main__':
    main()