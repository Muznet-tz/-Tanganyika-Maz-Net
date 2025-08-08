// Full Web App for Cow Biometric Identification using React + Flask backend with GPS, tracking, and management features

// === React Frontend (App.jsx) ===

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';

export default function TanganyikaMazNet() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [location, setLocation] = useState(null);
  const [nextVaccination, setNextVaccination] = useState('2025-08-20');
  const [waterNeed, setWaterNeed] = useState('40 Liters/day');
  const [foodNeed, setFoodNeed] = useState('25 kg/day');

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    setImage(URL.createObjectURL(file));
    setResult(null);
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  const handleTrackGPS = () => {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const coords = pos.coords;
        setLocation({ lat: coords.latitude, lon: coords.longitude });
      },
      (err) => console.error('GPS Error:', err)
    );
  };

  return (
    <div className="p-6 grid gap-4">
      <h1 className="text-2xl font-bold">Tanganyika Maz Net</h1>
      <Input type="file" accept="image/*" onChange={handleUpload} />
      {image && (
        <Card>
          <CardContent>
            <img src={image} alt="Nose Print" className="w-full max-w-md mt-2" />
          </CardContent>
        </Card>
      )}
      {loading && <p>Identifying cow...</p>}
      {result && (
        <div className="mt-4 text-lg">
          <p>Identified Cow ID: <strong>{result.cow_id}</strong></p>
          <p>Confidence: {Math.round(result.confidence * 100)}%</p>
        </div>
      )}

      <Button onClick={handleTrackGPS}>Track Cow Location</Button>
      {location && (
        <p>Current GPS: {location.lat.toFixed(5)}, {location.lon.toFixed(5)}</p>
      )}

      <div className="mt-4">
        <h2 className="text-xl font-semibold">Cow Management Info</h2>
        <p>Next Vaccination: {nextVaccination}</p>
        <p>Daily Water Need: {waterNeed}</p>
        <p>Daily Food Need: {foodNeed}</p>
      </div>
    </div>
  );
}

// === Flask Backend (app.py) ===

# Save this in a separate Python file (e.g., app.py)

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('model.h5')

CLASS_NAMES = ['Cow001', 'Cow002', 'Cow003', 'Cow004', 'Cow005']  # Update based on your dataset

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L').resize((128, 128))
    image = np.array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = preprocess_image(file.read())
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    return jsonify({
        'cow_id': CLASS_NAMES[predicted_class],
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)
