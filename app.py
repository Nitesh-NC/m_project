from flask import Flask, render_template, request, Response
from PIL import Image
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import gdown

app = Flask(__name__)

# Define paths for the two models on Google Drive
model1_drive_link = 'https://drive.google.com/file/d/1982Kkm4-QJkx-CTloDoo4kgiKpB9q5jc/view?usp=sharing'
model2_drive_link = 'https://drive.google.com/file/d/1rgU38oEotPSwgeRlzDLHz8Q5r3WszlsC/view?usp=drive_link'

# Download models from Google Drive
gdown.download(model1_drive_link, 'New_Eye_model.h5', quiet=False)
gdown.download(model2_drive_link, 'eyediseases_outer_model.h5', quiet=False)

# Load the two models
model1 = tf.keras.models.load_model("New_Eye_model.h5")
model2 = tf.keras.models.load_model("eyediseases_outer_model.h5")

class_labels_model1 = {
    0: 'AMD',
    1: 'Cataract',
    2: 'Glaucoma',
    3: 'Hypertension',
    4: 'Myopia',
    5: 'noneye',
    6: 'Normal'
}

class_labels_model2 = {
    0: 'Normal',
    1: 'Cataract_O',
    2: 'Glaucoma_O',
    3: 'Infected_O',
    4: 'noneye',
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

def predict_disease(model, img_array, class_labels):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return class_labels[predicted_class]

def get_disease_info(prediction):
    disease_info = {
        'Normal': {
            'message': 'Your eyes are healthy.'
        },
        'Hypertension':{
            'cause': 'Often due to lifestyle factors, genetic predisposition, or underlying health conditions that lead to elevated blood pressure.',
            'major_reason': 'The major reason is increased pressure in the arteries, putting strain on the heart and potentially causing damage to blood vessels.',
            'prevention': 'Maintain a healthy lifestyle with a balanced diet, regular exercise, limited salt intake, and regular blood pressure monitoring. Avoid smoking and excessive alcohol consumption.',
            'treatment': 'Treatment may include medication, lifestyle modifications, and managing underlying health issues. Regular check-ups are crucial for monitoring and adjusting the treatment plan.',
            'stage': 'Stages range from mild to severe. Early detection and management are essential to prevent complications such as heart disease, stroke, or kidney damage.'
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
        'Cataract_O': {
            'cause': "Caused by the clouding of the eye's natural lens due to aging, injury, or other medical conditions.",
            'major_reason': 'Age is the primary risk factor, but other factors include exposure to UV light, smoking, and diabetes.',
            'prevention': 'Protect eyes from UV rays, quit smoking, manage diabetes, and have regular eye check-ups.',
            'treatment': 'Surgery to remove the cloudy lens and replace it with an artificial one.',
            'stage': 'Progresses from early symptoms like blurry vision to advanced stages with significant vision impairment.'
        },
        'Glaucoma_O': {
            'cause': 'Caused by increased intraocular pressure, leading to damage of the optic nerve.',
            'major_reason': 'High intraocular pressure, age, family history, and certain medical conditions contribute to the risk.',
            'prevention': 'Regular eye check-ups, early detection, and treatment can help prevent vision loss.',
            'treatment': 'Eye drops, oral medications, laser therapy, or surgery, depending on the severity.',
            'stage': 'Often asymptomatic in the early stages, progressing to vision loss if untreated.'
        },
        'Infected_O':{
            'cause': 'Caused by bacteria, viruses, fungi, or parasites from contaminated sources or contact with an infected person.',
            'major_reason': 'Introduction of harmful microorganisms into the eye, leading to inflammation and irritation.',
            'prevention': 'Practice good hygiene, avoid touching eyes with dirty hands, and use protective eyewear. Seek prompt medical attention for any symptoms.',
            'treatment': 'May include antibiotic/antiviral eye drops, warm compresses, and oral medications. Severe cases require medical intervention.',
            'stage': 'Progresses from mild redness to increased discomfort, discharge, and potential vision problems. Early medical attention is crucial.'
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
            'message': 'Please enter valid Image.'
        }
    }

    return disease_info.get(prediction, {})


# Define two routes for each model
@app.route('/')
def index():
    return render_template('frontpage.html')

@app.route('/model1', methods=['POST'])
def upload_file_model1():
    return process_image(model1, class_labels_model1)

@app.route('/model2', methods=['POST'])
def upload_file_model2():
    return process_image(model2, class_labels_model2)

def process_image(model, class_labels):
    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        img = Image.open(uploaded_file)
        img_array = preprocess_image(img)
        prediction = predict_disease(model, img_array, class_labels)
        disease_info = get_disease_info(prediction)

        return render_template('output.html', prediction=prediction, disease_info=disease_info)
    else:
        return render_template('frontpage.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
