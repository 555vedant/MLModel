from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import sklearn
# Load the trained model
model = pickle.load(open('model2.pkl', 'rb'))

# List of possible classes
severity_levels = ['Fatal', 'Grievous Injury', 'Damage Only', 'Simple Injury', 'ACHARI']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_accident():
    # Extract input data from the JSON request
    data = request.json
    Weather = int(data['weather'])
    District = int(data['district'])
    NumberOfVehicles = int(data['numberOfVehicles'])
    Latitude = float(data['latitude'])
    Longitude = float(data['longitude'])

    # Make prediction
    result = model.predict(np.array([Weather, District, NumberOfVehicles, Latitude, Longitude]).reshape(1, -1))

    # Get the predicted class from the severity_levels list
    prediction = severity_levels[result[0]]

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
