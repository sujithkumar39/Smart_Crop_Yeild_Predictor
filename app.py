import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------
# TRAIN MODEL ONLY ONCE
# -----------------------------------------------------------
@st.cache_resource
def train_model():

    df = pd.read_csv("data/yield_df.csv")

    # Remove unwanted columns
    df = df.drop(columns=["Year"], errors="ignore")

    # Encoders
    le_area = LabelEncoder()
    le_item = LabelEncoder()
    df["Area_enc"] = le_area.fit_transform(df["Area"])
    df["Item_enc"] = le_item.fit_transform(df["Item"])

    # Features
    X = df[["Area_enc", "Item_enc", "average_rain_fall_mm_per_year",
            "pesticides_tonnes", "avg_temp"]]
    y = df["hg/ha_yield"]

    # Scale numeric
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]] = \
        scaler.fit_transform(X_scaled[["average_rain_fall_mm_per_year",
                                       "pesticides_tonnes", "avg_temp"]])

    # Train
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=250, random_state=42)
    model.fit(X_train, y_train)

    return model, le_area, le_item, scaler

# Get trained components
model, le_area, le_item, scaler = train_model()

# -----------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------

st.title("🌾 Smart Crop Yield Predictor – No PKL Files Needed")

st.sidebar.header("Inputs")

area = st.sidebar.selectbox("Select Area", le_area.classes_)
item = st.sidebar.selectbox("Select Crop", le_item.classes_)

rainfall = st.sidebar.slider("Rainfall (mm)", 0, 3000, 1000)
pesticides = st.sidebar.slider("Pesticides (tonnes)", 0.0, 1000.0, 50.0)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)

if st.sidebar.button("Predict"):

    # Encode input
    area_enc = le_area.transform([area])[0]
    item_enc = le_item.transform([item])[0]

    # Create frame
    X_input = np.array([[area_enc, item_enc, rainfall, pesticides, temperature]])
    X_input[:, 2:] = scaler.transform(X_input[:, 2:])

    # Predict
    pred = model.predict(X_input)[0]

    st.success(f"🌱 Predicted Yield: **{pred:.2f} hg/ha**")

    st.write({
        "Area": area,
        "Crop": item,
        "Rainfall": rainfall,
        "Pesticides": pesticides,
        "Temperature": temperature
    })
