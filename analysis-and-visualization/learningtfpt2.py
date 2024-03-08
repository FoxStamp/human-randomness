import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("fashion_mnist.h5")

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(test_images[0])
print(type(test_images[0]))

prediction = model.predict(np.expand_dims(test_images[0], axis=0))

predictions = model.predict(test_images)