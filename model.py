# model.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------
# TRAIN MODEL WHEN THIS FILE IS IMPORTED
# -----------------------------------------------------------

def train_model():
    df = pd.read_csv("data/yield_df.csv")

    # Drop waste column
    df = df.drop(columns=["Year"], errors="ignore")

    # Label Encoding
    le_area = LabelEncoder()
    le_item = LabelEncoder()

    df["Area_enc"] = le_area.fit_transform(df["Area"])
    df["Item_enc"] = le_item.fit_transform(df["Item"])

    # Features and Target
    X = df[[
        "Area_enc",
        "Item_enc",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp"
    ]]
    y = df["hg/ha_yield"]

    # Scaling numerical columns
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[["average_rain_fall_mm_per_year",
              "pesticides_tonnes",
              "avg_temp"]] = scaler.fit_transform(
        X_scaled[["average_rain_fall_mm_per_year",
                  "pesticides_tonnes",
                  "avg_temp"]]
    )

    # Train model
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_scaled, y)

    return model, le_area, le_item, scaler


# Train model ONCE, reuse across app
model, le_area, le_item, scaler = train_model()


# -----------------------------------------------------------
# PREDICTION FUNCTION (Used by app.py)
# -----------------------------------------------------------

def predict_yield(area, item, rainfall, pesticides, temp):
    # Transform categorical values
    area_enc = le_area.transform([area])[0]
    item_enc = le_item.transform([item])[0]

    # Create feature row
    X = [[area_enc, item_enc, rainfall, pesticides, temp]]

    # Scale numerical values
    X_scaled = X.copy()
    X_scaled = scaler.transform(X)

    # Predict
    return model.predict(X_scaled)[0]
