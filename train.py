import glob
import keras.utils
import numpy as np
from keras.preprocessing import image
import scipy
import os
from keras.applications.imagenet_utils import preprocess_input

from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.applications.resnet50 import *
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


epoch = 10

def load_data(dir):
    img_data_list = []
    y = []
    if not os.path.exists(dir):
        print("Unable to load: " + dir)
    dir = dir + "*.jpg"
    filelist = glob.glob(dir)
    count = 0
    for fname in filelist:
        img = image.load_img(fname, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        cl,name = (os.path.basename(fname)).split("_")
        print('Input image shape:', x.shape, fname, cl)
        img_data_list.append(x)
        y.append(int(cl))
        if(count==500):
            break
        count = count+1
    return img_data_list, y

img_data_list, cl = load_data("Food/validation/")
img_data = np.array(img_data_list)
cl = np.array(cl)
print (cl.shape)
img_data=np.rollaxis(img_data,1,0)
img_data=img_data[0]
print (img_data.shape)

num_classes = 11

names = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"]

Y = keras.utils.to_categorical(cl, num_classes)

#shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

# Custom_resnet_model_1
#Training the classifier alone
image_input = Input(shape=(224, 224, 3))

model = ResNet50(input_tensor=image_input, include_top=False,weights='imagenet')
#model.load_weights('weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5py')

model.summary()

last_layer = model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
# a softmax layer for 4 classes
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
custom_resnet_model2 = Model(inputs=model.input, outputs=out)

custom_resnet_model2.summary()

for layer in custom_resnet_model2.layers[:-6]:
        layer.trainable = False

custom_resnet_model2.layers[-1].trainable

custom_resnet_model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

custom_resnet_model2.fit(X_train, y_train, batch_size=32, epochs=epoch, verbose=1, validation_data=(X_test, y_test))

(loss, accuracy) = custom_resnet_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

modelpath = "weights/cu_resnet.h5"
custom_resnet_model2.save(modelpath)

model1 = keras.models.load_model(modelpath)

img = image.load_img('Food/validation/4_211.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model1.predict(x)
import operator
index, value = max(enumerate(max(preds)), key=operator.itemgetter(1))
print('Predicted:', names[index], value)
