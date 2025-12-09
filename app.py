from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


import joblib

# Load both model and label encoder from the tuple
model, le = joblib.load('svm_crop_pipeline.pkl')





def fetch_location_name(lat, lng):
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {'format': 'json', 'lat': lat, 'lon': lng, 'zoom': 18, 'addressdetails': 1}
        headers = {'User-Agent': 'GeoSenseAI/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        address = data.get('address', {})
        location_name = (
            address.get('village') or 
            address.get('town') or 
            address.get('city') or 
            address.get('municipality') or
            address.get('suburb') or
            address.get('neighbourhood') or
            address.get('county') or
            address.get('state_district') or
            address.get('state') or
            address.get('country') or
            'Unknown Location'
        )
        if location_name in [address.get('state'), address.get('country')]:
            parts = data.get('display_name', '').split(',')
            if parts:
                location_name = parts[0].strip()
        return location_name
    except Exception as e:
        print(f"Error fetching location: {e}")
        return 'Unknown Location'


import requests
import numpy as np
from datetime import datetime, timedelta

import requests
from datetime import datetime, timedelta

def fetch_weather_data(lat, lng):
    try:
        # Step 1: Fetch current temperature & humidity (forecast for today)
        url_current = "https://api.open-meteo.com/v1/forecast"
        params_current = {
            'latitude': lat,
            'longitude': lng,
            'hourly': 'temperature_2m,relative_humidity_2m',
            'timezone': 'auto'
        }
        response_current = requests.get(url_current, params=params_current, timeout=10)
        data_current = response_current.json()
        hourly = data_current.get('hourly', {})
        temperature = round(sum(hourly.get('temperature_2m', [25.0])) / max(len(hourly.get('temperature_2m', [1])), 1), 2)
        humidity = round(sum(hourly.get('relative_humidity_2m', [65.0])) / max(len(hourly.get('relative_humidity_2m', [1])), 1), 2)

        # Step 2: Fetch annual rainfall from Historical API
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=365)
        url_rain = "https://archive-api.open-meteo.com/v1/archive"
        params_rain = {
            'latitude': lat,
            'longitude': lng,
            'daily': 'precipitation_sum',
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'timezone': 'auto'
        }
        response_rain = requests.get(url_rain, params=params_rain, timeout=10)
        data_rain = response_rain.json()
        daily_rain = data_rain.get('daily', {}).get('precipitation_sum', [])
        rainfall = round(sum(daily_rain)/10, 2) if daily_rain else 0.0

        return {'temperature': temperature, 'humidity': humidity, 'rainfall': rainfall}

    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return {'temperature': 25.0, 'humidity': 65.0, 'rainfall': 1200.0}




def fetch_soil_data(lat, lng):
    try:
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lng}&lat={lat}&property=nitrogen,phh2o,cec&depth=0-5cm&value=mean"
        response = requests.get(url, timeout=10)
        data = response.json()

        nitrogen, phosphorus, potassium, ph = 0.0, 0.0, 0.0, 7.0
        if 'properties' in data and 'layers' in data['properties']:
            for layer in data['properties']['layers']:
                if layer['name'] == 'nitrogen':
                    nitrogen = round(layer['depths'][0]['values']['mean'], 2)
                elif layer['name'] == 'phh2o':
                    ph = round(layer['depths'][0]['values']['mean'], 2)
                elif layer['name'] == 'cec':
                    cec_value = layer['depths'][0]['values']['mean']
                    phosphorus = round(cec_value * 0.8, 2)
                    potassium = round(cec_value * 2.5, 2)

        if nitrogen == 0.0: nitrogen = round(np.random.uniform(10, 50), 2)
        if phosphorus == 0.0: phosphorus = round(np.random.uniform(20, 80), 2)
        if potassium == 0.0: potassium = round(np.random.uniform(100, 300), 2)

        return {'N': nitrogen, 'P': phosphorus, 'K': potassium, 'ph': ph}
    except Exception as e:
        print(f"Error fetching soil data: {e}")
        return {'N': round(np.random.uniform(20, 40), 2),
                'P': round(np.random.uniform(30, 70), 2),
                'K': round(np.random.uniform(150, 250), 2),
                'ph': round(np.random.uniform(6.5, 7.5), 2)}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or le is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Get coordinates from request
        data = request.get_json()
        lat = float(data['lat'])
        lng = float(data['lng'])

        # Fetch location name
        location_name = fetch_location_name(lat, lng)

        # Fetch weather and soil data
        weather = fetch_weather_data(lat, lng)
        soil = fetch_soil_data(lat, lng)

        # Prepare features in the same order used during training
        features = np.array([[soil['N'], soil['P'], soil['K'],
                              weather['temperature'], weather['humidity'],
                              soil['ph'], weather['rainfall']]])

        # Predict numeric label
        numeric_pred = model.predict(features)[0]

        # Convert numeric label to crop name
        crop_name = le.inverse_transform([numeric_pred])[0]

        # Return JSON response
        return jsonify({
            'prediction': str(crop_name),
            'location_name': location_name,
            'features': {
                'N': soil['N'],
                'P': soil['P'],
                'K': soil['K'],
                'ph': soil['ph'],
                'temperature': weather['temperature'],
                'humidity': weather['humidity'],
                'rainfall': weather['rainfall']
            }
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
