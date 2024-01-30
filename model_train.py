from keras.models import Model
from keras.layers import Input, Dense
import pandas as pd
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('iris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# Encoding labels with scikit-learn LabelEncoder
label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = to_categorical(data['class'])

# Split the data into training and testing sets using scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

inp = Input(shape=(4))
x = Dense(32, activation="relu")(inp)
op = Dense(3, activation="softmax")(x)

model = Model(inputs=inp, outputs=op)

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['acc'])
model.fit(X_train, y_train, epochs=30)

model.save('model.h5')

arr = label_encoder.classes_

np.save("labels.npy", arr)
