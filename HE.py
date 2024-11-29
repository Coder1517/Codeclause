#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st


# In[4]:


data = pd.read_csv("heart.csv")

# Define features and target variable
X = data.drop(columns=['target'])  # Independent variables
y = data['target']                 # Dependent variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the model
with open("heart_disease_model.pkl", "wb") as file:
    pickle.dump(model, file)


# In[5]:


# Load the trained model
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Heart Disease Risk Prediction")

# Input fields for user data
age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
chol = st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise-Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (1-3)", [1, 2, 3])

# Prepare data for prediction
input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal]])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    risk = "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk of Heart Disease"
    st.write(f"Prediction: {risk}")


# In[ ]:




