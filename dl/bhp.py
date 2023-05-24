import numpy as np
import pandas as pd
import tensorflow as tf

# Load the Boston housing dataset
boston_data = pd.read_csv('/content/Boston.csv')

boston_data.head()

# Perform exploratory data analysis
boston_data.describe()
boston_data.hist()

# Identify the features and target variable
features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
target = 'medv'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(boston_data[features], boston_data[target], test_size=0.2)

# Create a neural network model with one hidden layer
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100)

# Evaluate the model on the testing set
model.evaluate(X_test, y_test)

# Make predictions on new data
predictions = model.predict(X_test)

for i in range(10):
    print('Prediction:', predictions[i][0], 'Actual:', y_test.iloc[i])



