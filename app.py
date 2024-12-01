from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
import os

# Charger le modèle
model = tf.keras.models.load_model("handsign_recognition_model.h5")
categories = ["pituitary", "notumor", "meningioma", "glioma"]

# Créer l'application Flask
app = Flask(__name__)

# Route pour la page principale
@app.route('/')
def index():
    return render_template('index.html')

# Route pour faire une prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Lire et prétraiter l'image
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img / 255.0, axis=0)

    # Prédiction
    predictions = model.predict(img)
    predicted_class = categories[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return jsonify({"class": predicted_class, "confidence": confidence})

# Route pour servir les fichiers statiques
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# Ensure the app binds to the correct host and port when deployed on Render
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
