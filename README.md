# Foodie

Foodie is a web app for detection of food/beverages from an image.
Thousands of people post their food pics on social media daily. Automatic identification of food types from the pictures can be very useful on many occasions.
A convolutional neural network is trained over Resnet on a dataset containing 11 categories of food viz. "Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit".
A web application environment is setup using flask micro web framework to provide a simple GUI.

##Usage:
Clone the model.
Download pretrained weights for Resnet on imagenet dataset.
Train the model:
```bash
$ python3 train.py
```
Running the the web app
```bash
$ export FLASK_APP=predict_app.py
$ flask run --host=0.0.0.0
```
Open http://0.0.0.0:5000/static/predict.html from browser to see the application running on social machine.

##Screenshot view
![screenshot](https://github.com/Subarno/Foodie/blob/master/static/Screenshot%20from%202018-06-10%2001-23-46.png)
