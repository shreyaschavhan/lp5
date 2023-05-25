# -*- coding: utf-8 -*-
"""mnist-fashion-dataset-dl-assignment-2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pKxAtSz_4R4UKewdD2YF32igKFS6NriS
"""

import tensorflow as tf
import numpy as np

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create the model
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
model.evaluate(x_test, y_test)

# Make predictions
predictions = model.predict(x_test)

predicted_labels = np.argmax(np.round(predictions), axis=1)

correct = np.where(predicted_labels==y_test)[0]
print("Correct labels: ", len(correct))

import matplotlib.pyplot as plt
for i, correct in enumerate(correct[:9]):
  plt.subplot(3,3,i+1)
  plt.imshow(x_test[correct].reshape(28,28))
  plt.title("Predicted  {}, Class {}".format(predicted_labels[correct], y_test[correct]))

