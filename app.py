import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.title("🌊 Sonar Signal Classifier")
st.write("Classify underwater acoustic return signals as a Rock or a Mine using a Support Vector Machine model.")

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("sonar_data.csv", header=None)
    df[60] = df[60].map({"R": 0, "M": 1})
    
    X = df.drop(columns=[60])
    y = df[60]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    return model, scaler, acc, df

try:
    model, scaler, accuracy, df = load_and_train_model()
    st.write(f"Model Training Accuracy: {accuracy * 100:.2f}%")
except Exception as e:
    st.write(f"Error loading `sonar_data.csv`. Ensure the file is in the working directory. Details: {e}")
    st.stop()

st.divider()

st.write("### Test Single Signal")
st.write("Select a row index from the dataset to test prediction:")

row_idx = st.number_input("Enter Row Index", min_value=0, max_value=len(df)-1, value=0, step=1)

st.divider()

if st.button("Predict Object Type"):
    sample_features = df.drop(columns=[60]).iloc[row_idx].values
    actual_label = df.iloc[row_idx][60]
    actual_text = "Mine (M)" if actual_label == 1 else "Rock (R)"
    
    input_array = sample_features.reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    pred_text = "⚠️ MINE" if prediction == 1 else "🪨 ROCK"
    
    st.write(f"Actual Object Type: {actual_text}")
    st.write(f"Prediction Result: {pred_text}")
    st.write(f"Mine Probability: {probabilities[1]*100:.2f}%")
    st.write(f"Rock Probability: {probabilities[0]*100:.2f}%")