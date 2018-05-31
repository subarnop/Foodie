import numpy as np
from PIL import Image
from keras.preprocessing import image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
import cv2 as cv


def test(img,model):

    names = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"]
    img = image.load_img(img, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(x.shape)
    preds = model.predict(x)
    import operator
    index, value = max(enumerate(max(preds)), key=operator.itemgetter(1))
    print('Predicted:', names[index], value)

    '''
    print(img)
    prediction = "Hello"
    print(prediction)'''
    return names[index], value

#print(test("0_14.jpg"))
