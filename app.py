import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

MODEL_PATH = 'model_artifacts/autoencoder_model.h5'
SCALER_PATH = 'model_artifacts/scaler.pkl'
THRESHOLD_PATH = 'model_artifacts/threshold.npy'


def load_resources():
    global model, scaler, threshold
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='mae')
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(THRESHOLD_PATH, 'rb') as f:
            threshold = np.load(f)
        print('Resources loaded successfully.')
    else:
        model = None
        scaler = None
        threshold = None
        print('Error: Model files not found. Run train_model.py first.')


load_resources()


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ''
    result_class = ''

    if request.method == 'POST' and scaler is not None and model is not None and threshold is not None:
        try:
            features = [
                float(request.form['engine_temperature_C']),
                float(request.form['oil_pressure_bar']),
                float(request.form['engine_rpm']),
                float(request.form['fuel_consumption_lph']),
                float(request.form['vibration_mm_s'])
            ]

            input_data = np.array([features])
            input_scaled = scaler.transform(input_data)
            reconstruction = model.predict(input_scaled)
            loss = np.mean(np.abs(input_scaled - reconstruction))

            if loss > threshold:
                prediction_text = f'ANOMALY DETECTED! (Loss: {loss:.4f} > Threshold: {threshold:.4f})'
                result_class = 'danger'
            else:
                prediction_text = f'Engine Normal. (Loss: {loss:.4f})'
                result_class = 'success'
        except Exception as exc:
            prediction_text = f'Error: {exc}'
            result_class = 'text-warning'
    elif request.method == 'POST':
        prediction_text = 'Model resources missing. Run training first.'
        result_class = 'text-warning'

    return render_template('index.html', prediction=prediction_text, result_class=result_class)


if __name__ == '__main__':
    app.run(debug=True)