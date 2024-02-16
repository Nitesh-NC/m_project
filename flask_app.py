from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

model_path = "anomaly_model.h5"
model = tf.keras.models.load_model(model_path)

class_labels = {
    0: 'Normal',
    1: 'Cataract',
    2: 'Glaucoma',
    3: 'AMD',
    4: 'Myopia',
    5: 'noneye',
}

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)

    # Ensure the image has 3 channels (RGB)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_disease(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return class_labels[predicted_class]

def get_disease_info(prediction):
    disease_info = {
        'Normal': {
            'message': 'Your eyes are healthy.'
        },
        'Cataract': {
            'cause': "Caused by the clouding of the eye's natural lens due to aging, injury, or other medical conditions.",
            'major_reason': 'Age is the primary risk factor, but other factors include exposure to UV light, smoking, and diabetes.',
            'prevention': 'Protect eyes from UV rays, quit smoking, manage diabetes, and have regular eye check-ups.',
            'treatment': 'Surgery to remove the cloudy lens and replace it with an artificial one.',
            'stage': 'Progresses from early symptoms like blurry vision to advanced stages with significant vision impairment.'
        },
        'Glaucoma': {
            'cause': 'Caused by increased intraocular pressure, leading to damage of the optic nerve.',
            'major_reason': 'High intraocular pressure, age, family history, and certain medical conditions contribute to the risk.',
            'prevention': 'Regular eye check-ups, early detection, and treatment can help prevent vision loss.',
            'treatment': 'Eye drops, oral medications, laser therapy, or surgery, depending on the severity.',
            'stage': 'Often asymptomatic in the early stages, progressing to vision loss if untreated.'
        },
        'AMD': {
            'cause': 'Caused by degeneration of the macula, the central part of the retina.',
            'major_reason': 'Age is the primary risk factor, along with genetic factors and smoking.',
            'prevention': 'Healthy lifestyle choices, including a balanced diet rich in antioxidants, can help reduce the risk.',
            'treatment': 'No cure, but certain medications or therapies may slow down the progression in some cases.',
            'stage': 'Early AMD may have no symptoms, while advanced stages can lead to central vision loss.'
        },
        'Myopia': {
            'cause': 'Caused by the elongation of the eyeball or excessive curvature of the cornea, leading to difficulty seeing distant objects.',
            'major_reason': 'Genetics play a significant role, and environmental factors like prolonged near work can contribute.',
            'prevention': 'Encourage outdoor activities, take breaks during near work, and have regular eye check-ups.',
            'treatment': 'Corrective lenses (glasses or contact lenses) or refractive surgery like LASIK.',
            'stage': 'Develops during childhood and progresses, stabilizing in adulthood.'
        },
        'noneye': {
            'message': 'This is not a fundus image. Please enter valid data.'
        }
    }

    return disease_info.get(prediction, {})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        img = Image.open(uploaded_file)
        img_array = preprocess_image(img)
        prediction = predict_disease(img_array)
        disease_info = get_disease_info(prediction)

        return render_template('result.html', prediction=prediction, disease_info=disease_info)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)