from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import os

app = Flask(__name__)

# Load the model and dictionary once when the application starts
model = tf.keras.models.load_model(
       ('./models/full-image-set-mobilenetv2-Adam.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer}
)
model.make_predict_function()

# Load breed labels from CSV into a dictionary
breeds = pd.read_csv('./Data/labels.csv')
breed_dict = dict(enumerate(breeds['breed'].unique()))

def predict_label(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = img.reshape(1, 224, 224, 3)
    prediction = model.predict(img)
    predicted_class = prediction.argmax()
    return breed_dict[predicted_class]

# Define routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Hello, Team FlashBolt..!!!"

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        
        # Create the directory if it doesn't exist
        if not os.path.exists(os.path.join(app.root_path, 'static')):
            os.makedirs(os.path.join(app.root_path, 'static'))
        
        img_path = os.path.join(app.root_path, 'static', img.filename)
        img.save(img_path)

        prediction = predict_label(img_path)

        return render_template("index.html", prediction=prediction, img_filename=img.filename)


if __name__ == '__main__':
    app.run(debug=True)
