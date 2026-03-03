import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# -----------------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------------
df = pd.read_csv("data/yield_df.csv")

# -----------------------------------------------------------
# 2. Drop Year (you said waste)
# -----------------------------------------------------------
df = df.drop(columns=["Year"], errors="ignore")

# -----------------------------------------------------------
# 3. Encode categorical columns
# -----------------------------------------------------------
le_area = LabelEncoder()
le_item = LabelEncoder()

df["Area_enc"] = le_area.fit_transform(df["Area"])
df["Item_enc"] = le_item.fit_transform(df["Item"])

# -----------------------------------------------------------
# 4. Define features (X) and target (y)
# -----------------------------------------------------------
X = df[["Area_enc", "Item_enc", "average_rain_fall_mm_per_year",
        "pesticides_tonnes", "avg_temp"]]

y = df["hg/ha_yield"]

# -----------------------------------------------------------
# 5. Scale numerical features
# -----------------------------------------------------------
scaler = StandardScaler()

X_scaled = X.copy()
X_scaled[["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]] = scaler.fit_transform(
    X_scaled[["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]]
)

# -----------------------------------------------------------
# 6. Train-test split
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 7. Train Model
# -----------------------------------------------------------
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------------------------------
# 8. Evaluate
# -----------------------------------------------------------
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)

# -----------------------------------------------------------
# 9. Save model and transformers
# -----------------------------------------------------------
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/crop_yield_model.pkl")
joblib.dump(le_area, "model/encoder_Area.pkl")
joblib.dump(le_item, "model/encoder_Item.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model, encoders, and scaler saved successfully!")