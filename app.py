import streamlit as st
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load and preprocess data using scikit-learn
def load_and_preprocess_data():
    data = pd.read_csv('iris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

    # Encoding labels with scikit-learn LabelEncoder
    label_encoder = LabelEncoder()
    data['class'] = label_encoder.fit_transform(data['class'])

    X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y = to_categorical(data['class'])

    return X, y, label_encoder.classes_

st.title("Welcome to flower prediction app")

# Load the model and labels
model = load_model("model.h5")
labels = np.load("labels.npy")

# Load and preprocess data using scikit-learn
X, y, classes = load_and_preprocess_data()

# Split the data into training and testing sets using scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

a = float(st.number_input("sepal length in cm"))
b = float(st.number_input("sepal width in cm"))
c = float(st.number_input("petal length in cm"))
d = float(st.number_input("petal width in cm"))

btn = st.button("Predict")

if btn:
    # Use the model to make predictions
    pred = model.predict(np.array([a, b, c, d]).reshape(1, -1))

    # Get the predicted label
    pred_label = labels[np.argmax(pred)]
    st.subheader(f"Predicted Label: {pred_label}")

    # Convert the predicted label to the corresponding class name
    predicted_class_name = classes[pred_label]

    # Display the corresponding image
    st.image(f"{predicted_class_name.lower()}.jpg", caption=f"{predicted_class_name} Iris")
