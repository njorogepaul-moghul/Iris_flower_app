import streamlit as st
import numpy as np
import pickle

st.title("🌸 Iris Flower Species Prediction App")

st.write("Adjust the sliders below to set the flower measurements and choose a model for prediction:")

# Sliders for 4 original features
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# --- Engineered features ---
petal_ratio = petal_length / petal_width if petal_width != 0 else 0
sepal_ratio = sepal_length / sepal_width if sepal_width != 0 else 0
petal_area = petal_length * petal_width
sepal_area = sepal_length * sepal_width
aspect_ratio = petal_length / sepal_length if sepal_length != 0 else 0
compactness = (sepal_width + petal_width) / (sepal_length + petal_length) if (sepal_length + petal_length) != 0 else 0

# Final feature vector (must match training order)
features = np.array([[sepal_length, sepal_width, petal_length, petal_width,
                      petal_ratio, sepal_ratio, petal_area, sepal_area,
                      aspect_ratio, compactness]])

# --- Model selection dropdown ---
model_choice = st.selectbox("Choose a model:", 
                            ["Logistic Regression", "Random Forest", "Support Vector Machine"])

# Load the chosen model 
if model_choice == "Logistic Regression":
    model = pickle.load(open("Iris_model.pkl", "rb"))
elif model_choice == "Random Forest":
    model = pickle.load(open("Iris_model_rf.pkl", "rb"))
else:
    model = pickle.load(open("Iris_svm_clf.pkl", "rb"))

# Predict button
if st.button("Predict"):
    prediction = model.predict(features)
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    species_name = species_map[int(prediction[0])]
    st.success(f"🌿 Predicted Iris Species using {model_choice}: **{species_name}**") 

