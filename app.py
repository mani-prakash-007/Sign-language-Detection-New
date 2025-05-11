import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Force TF backend

import cv2
import numpy as np
import tensorflow as tf
import keras
from flask import Flask, render_template, Response, jsonify
from keras.saving import load_model
import joblib
from PIL import Image
import platform

app = Flask(__name__)

# ================== KERAS 3 COMPATIBILITY ==================
def custom_input_layer(**kwargs):
    kwargs.pop('batch_shape', None)
    return keras.layers.InputLayer(**kwargs)

custom_objects = {
    'InputLayer': custom_input_layer,
    'DTypePolicy': keras.dtype_policies.Policy,
    'Adam': keras.optimizers.Adam,
    'relu': keras.activations.relu,
    'softmax': keras.activations.softmax,
    'GlorotUniform': keras.initializers.GlorotUniform,
    'Zeros': keras.initializers.Zeros
}
# ===========================================================

# ================== MODEL LOADING ==================
MODEL_PATH = 'Sign_Language_Model.keras'  # Must be Keras 3 format

try:
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    print("✅ Model loaded successfully!")
    model.summary()
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    exit()

# ================== LABEL ENCODER ==================
try:
    label_encoder = joblib.load('label_encoder.pkl')
except Exception as e:
    print(f"❌ Label encoder error: {e}")
    exit()

# ================== MAC CAMERA SETUP ==================
IS_MAC = platform.system() == 'Darwin'

def init_camera():
    if IS_MAC:
        for index in [0, 1, 2]:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                print(f"📷 Using camera index {index}")
                return cap
        print("❌ No camera found!")
        exit()
    else:
        return cv2.VideoCapture(0)

camera = init_camera()

# ================== IMAGE PROCESSING ==================
def preprocess_image(image):
    try:
        img = Image.fromarray(image).resize((64, 64))
        img_array = np.array(img).astype('float32') / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"🖼️ Preprocessing error: {e}")
        return None

# ================== VIDEO STREAMING ==================
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = frame.copy()
            
            processed = preprocess_image(rgb_frame)
            if processed is not None:
                preds = model.predict(processed, verbose=0)
                pred_class = np.argmax(preds, axis=1)
                sign = label_encoder.inverse_transform(pred_class)[0]
                confidence = np.max(preds) * 100
            else:
                sign = "Error"
                confidence = 0.0

            cv2.rectangle(display_frame, (100, 100), (400, 400), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Sign: {sign}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Confidence: {confidence:.1f}%", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', display_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        except Exception as e:
            print(f"🎥 Frame error: {e}")
            break

# ================== FLASK ROUTES ==================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Camera read failed'})

    try:
        roi = frame[100:400, 100:400]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        processed = preprocess_image(rgb_roi)
        
        if processed is None:
            return jsonify({'error': 'Image processing failed'})
            
        preds = model.predict(processed, verbose=0)
        pred_class = np.argmax(preds, axis=1)
        sign = label_encoder.inverse_transform(pred_class)[0]
        confidence = np.max(preds) * 100

        return jsonify({
            'sign': sign,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("\n🚀 Starting Sign Language Detection App")
    print(f"🍎 macOS: {IS_MAC}")
    print(f"🔧 TF: {tf.__version__}, Keras: {keras.__version__}")
    print(f"💻 GPUs: {tf.config.list_physical_devices('GPU')}")
    app.run(host='0.0.0.0', port=5000, debug=True)