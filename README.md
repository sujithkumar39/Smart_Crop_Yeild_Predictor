# Smart_Crop_Yeild_Predictor
A Machine Learning application to predict yield of yearly crop based on the Type of crop, Rainfall, Pesticides, Temperature.
Developed using Streamlit and CSS for frontend and RandomForest Model at Backend.

📁1. Dataset Overview

The dataset contains country-wise crop production details with several environmental and agricultural features.
Features used-
  Area – Country/Region where the crop is grown
  Item – Crop name (e.g., Rice, Wheat, Maize, Pulses)
  Year – (Removed later as it’s not meaningful for prediction)
  average_rain_fall_mm_per_year – Annual average rainfall
  pesticides_tonnes – Pesticide use in tonnes
  avg_temp – Annual average temperature
  hg/ha_yield – Target variable (Yield per hectare)

🧹2. Data Preprocessing

Preprocessing included:

  Removing unnecessary columns - Year column removed because it does not influence yield directly.
  Handling missing values - The dataset had no null values, so no imputation required.
  
  Label Encoding-
  Categorical fields encoded using LabelEncoder: Area, Item
  
  Feature Scaling-
  Continuous variables scaled using StandardScaler: Rainfall, Pesticides, Temperature

🧪3. Feature Engineering

Features used for model training:
  
  Feature	Type-
    Area (encoded)	Categorical → numeric
    Item (encoded)	Categorical → numeric
    Rainfall	Continuous
    Pesticides	Continuous
    Temperature	Continuous
    
  Target variable: hg/ha_yield 

🤖 4. Model Used: Random Forest Regressor

  Reason for choosing RandomForestRegressor:
    Works well for non-linear relationships
    Robust to outliers
    Handles high variance in agricultural data
    Provides stable predictions
    Requires minimal parameter tuning
  
  Model saved as: crop_yield_model.pkl
  
🔀 5. Train–Test Split
  
  The dataset was split into: 
    80% → Training set
    20% → Testing set

🏋️ 6. Model Training

  The RandomForest model was trained on the processed features.
  It learns:
    How temperature affects crop growth
    How rainfall impacts productivity
    The effect of pesticides on soil and yield
    Area & crop-specific trends
    
  After training:
    Model stored in /model/ folder
    Encoders & scaler also saved for prediction time
  
📊 7. Output & Risk-Adjusted Prediction
  
  When a user enters values: 
    Inputs are encoded + scaled
    Sent to Random Forest model
    Prediction is generated
  
  A risk factor is applied based on:
    Excess rainfall
    Toxic pesticide levels
    Extreme temperatures    

🖥️8. Frontend (Streamlit + Custom CSS)

  Built an interactive frontend using Streamlit with:
  User input sliders & dropdowns
  Prediction card
  Risk alert styling
  Fully customized UI using embedded CSS    
