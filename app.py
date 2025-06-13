from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # ✅ Import this
from src.prediction import XrayClassifierModel, BrainDiseaseModel, ChestDiseaseModel
from src.disease_name import DiseaseInfo
import os

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Optionally restrict to certain origins (safer for production):
# CORS(app, ggg resources={r"/predict": {"origins": "http://localhost:3000"}})

# Load models
xray_model = XrayClassifierModel(
    model_path='models/brain_chest_resnet152_model_state_dict.pth',
    classnames_path='classnames/brain_chest.txt'
)

brain_model = BrainDiseaseModel(
    model_path='models/brain_mobilenetv2_model_state_dict.pth',
    classnames_path='classnames/brain.txt'
)

chest_model = ChestDiseaseModel(
    model_path='models/resnet18_chest.pth',
    classnames_path='classnames/chest.txt'
)

disease_info = DiseaseInfo(json_path='data.json')

@app.route('/')
def home():
    return jsonify({'message': 'X-ray Disease Prediction API Running'})

@app.route('/frontend')
def frontend():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image_bytes = file.read()

    predicted_category = xray_model.predict(image_bytes)

    if predicted_category.lower() == 'brain':
        predicted_class = brain_model.predict(image_bytes)
    # elif predicted_category.lower() == 'chest':
    #     predicted_class = chest_model.predict(image_bytes)
    else:
        return jsonify({'error': 'Unknown category predicted'}), 400

    info = disease_info.get_info(predicted_class)
    if info:
        return jsonify({
            'predicted_category': predicted_category,
            'predicted_class': predicted_class,
            'details': info
        })
    else:
        return jsonify({'error': 'Disease information not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
