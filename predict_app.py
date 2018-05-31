import base64
import numpy as np
from keras.models import load_model
import tensorflow as tf
import io
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from flask import request
from flask import jsonify
from flask import Flask
from keras.applications.imagenet_utils import preprocess_input

g = tf.Graph()
with g.as_default():
    print(" * Model loading")
    model = load_model('weights/cu_resnet.h5')

app = Flask(__name__)

@app.route("/predict", methods=["POST"])

def predict():
    message = request.get_json(force=True)
    img = message['image']
    '''
    decoded = base64.b64decode(encoded)
    #image = Image.open(io.BytesIO(decoded))
    '''
    #img = image.load_img(encoded, target_size=(224, 224))
    names = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"]
    img = image.load_img(img, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(x.shape)
    with g.as_default():
        preds = model.predict(x)
    import operator
    index, value = max(enumerate(max(preds)), key=operator.itemgetter(1))
    print('Predicted:', names[index], value)
    prediction, value = names[index], value
    print(prediction)
    response = {
        'prediction': {
            'prediction': prediction,
            'value' : str(value)
        }
    }
    return jsonify(response)
