import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def load_and_train_model():

    # 1. Load Dataset
    df = pd.read_csv("yield_df.csv")

    # 2. Drop Year (waste column)
    df = df.drop(columns=["Year"], errors="ignore")

    # 3. Label Encoding
    le_area = LabelEncoder()
    le_item = LabelEncoder()

    df["Area_enc"] = le_area.fit_transform(df["Area"])
    df["Item_enc"] = le_item.fit_transform(df["Item"])

    # 4. Features and Target
    X = df[[
        "Area_enc",
        "Item_enc",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp"
    ]]

    y = df["hg/ha_yield"]

    # 5. Scaling numerical features
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]] = \
        scaler.fit_transform(X_scaled[["average_rain_fall_mm_per_year",
                                       "pesticides_tonnes",
                                       "avg_temp"]])

    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 7. Train Model
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    return model, le_area, le_item, scaler
