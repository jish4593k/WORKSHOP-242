import sys
import falcon
from wsgiref import simple_server
from mongoengine import connect
import tensorflow as tf
from tensorflow import keras
import turtle

from app.resources.human import CustomHuman as HumanResource
from app.resources.human import CustomHumans as HumansResource
from app.middlewares.custom_json import CustomJSON
from app.middlewares.custom_models import CustomModels

# Connect to MongoDB
connect('falcon-workshop')


api = falcon.API(middleware=[
    CustomJSON(),
    CustomModels()
])


api.add_route('/entities/{id}', HumanResource())
api.add_route('/entities', HumansResource())


model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def draw_square():
    for _ in range(4):
        turtle.forward(100)
        turtle.right(90)

if __name__ == '__main__':

    httpd = simple_server.make_server('0.0.0.0', 8000, api)

   
    dummy_data = tf.random.normal((100, 10))
    dummy_labels = tf.random.randint(2, size=(100, 1))
    model.fit(dummy_data, dummy_labels, epochs=5)
    
    turtle.speed(2)
    draw_square()
    turtle.done()

    httpd.serve_forever()
