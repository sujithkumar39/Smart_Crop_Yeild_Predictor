import streamlit as st
import numpy as np
import joblib

# ------------------------------------------------------
# Load model + encoders + scaler
# ------------------------------------------------------
model = joblib.load("model/crop_yield_model.pkl")
le_area = joblib.load("model/encoder_Area.pkl")
le_item = joblib.load("model/encoder_Item.pkl")
scaler = joblib.load("model/scaler.pkl")

# ------------------------------------------------------
# App Configuration
# ------------------------------------------------------
st.set_page_config(
    page_title="🌾 Smart Crop Yield Predictor",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8fff1; }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        padding: 10px 22px;
        border-radius: 10px;
        font-size: 18px;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background: #ffffff;
        box-shadow: 0px 5px 15px rgba(0,0,0,0.1);
    }
    .tag-low {
        background: #ffcccc;
        padding: 6px 10px;
        border-radius: 8px;
        color: #a10000;
        font-weight: bold;
    }
    .tag-medium {
        background: #fff4cc;
        padding: 6px 10px;
        border-radius: 8px;
        color: #a68000;
        font-weight: bold;
    }
    .tag-high {
        background: #d1ffcc;
        padding: 6px 10px;
        border-radius: 8px;
        color: #006b1b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------------
st.sidebar.header("🌱 Input Controls")
area = st.sidebar.selectbox("Select Area", le_area.classes_)
item = st.sidebar.selectbox("Select Crop", le_item.classes_)
rainfall = st.sidebar.slider("Average Rainfall (mm per year)", 0, 3000, 1000)
pesticides = st.sidebar.slider("Pesticides (tonnes)", 0.0, 1000.0, 50.0)
temperature = st.sidebar.slider("Average Temperature (°C)", 0.0, 50.0, 25.0)

st.title("🌾Crop Yield Prediction System")
st.markdown("### Predict crop yield and understand climate impact.")

st.divider()


# ------------------------------------------------------
# AGRICULTURAL WARNINGS + RISK LOGIC
# ------------------------------------------------------
warnings = []
risk_factor = 0.0   # this is used to reduce predicted yield

# ————— RAINFALL —————
if rainfall > 2500:
    warnings.append("💧 **Dangerously high rainfall** → flooding, root rot, fungal attack.")
    risk_factor += 0.40
elif rainfall > 1800:
    warnings.append("💧 **Very high rainfall** → waterlogging & nutrient leaching.")
    risk_factor += 0.25
elif rainfall < 300:
    warnings.append("💧 **Severe drought risk** → stunted growth, flower drop.")
    risk_factor += 0.30
elif rainfall < 600:
    warnings.append("💧 **Low rainfall** → irrigation needed.")
    risk_factor += 0.15

# ————— PESTICIDES —————
if pesticides > 300:
    warnings.append("🧪 **Toxic pesticide levels** → soil damage, microbial loss.")
    risk_factor += 0.35
elif pesticides > 150:
    warnings.append("🧪 **High pesticide usage** → soil fertility decreases.")
    risk_factor += 0.20
elif pesticides < 1:
    warnings.append("🧪 **Too low pesticide usage** → pest attack likely.")
    risk_factor += 0.12

# ————— TEMPERATURE —————
if temperature > 42:
    warnings.append("🔥 **Extreme heat (>42°C)** → crop drying, sterility, major losses.")
    risk_factor += 0.45
elif temperature > 35:
    warnings.append("🔥 **High heat stress** → photosynthesis drops, grains shrink.")
    risk_factor += 0.25
elif temperature < 12:
    warnings.append("❄️ **Cold stress** → delayed flowering, poor growth.")
    risk_factor += 0.20


# ------------------------------------------------------
# Prepare Input
# ------------------------------------------------------
area_enc = le_area.transform([area])[0]
item_enc = le_item.transform([item])[0]

input_data = np.array([[area_enc, item_enc, rainfall, pesticides, temperature]])
input_data[:, 2:] = scaler.transform(input_data[:, 2:])

# ------------------------------------------------------
# Predict Button
# ------------------------------------------------------
if st.sidebar.button("🔍 Predict Yield"):

    prediction = model.predict(input_data)[0]

    # apply risk penalty
    prediction = prediction * (1 - risk_factor)
    prediction = max(prediction, 100)  # prevent negative/zero values

    # -------- Risk-Aware Yield Tag ----------
    try:
        # Adjust thresholds based on risk
        if risk_factor > 0.60:      # EXTREME risk
            low_th = 10000
            med_th = 20000
        elif risk_factor > 0.30:    # HIGH risk
            low_th = 7000
            med_th = 14000
        else:                       # LOW–MODERATE risk
            low_th = 2000
            med_th = 5000

        # Apply new thresholds
        if prediction < low_th:
            tag = '<span class="tag-low">LOW YIELD</span>'
            color = "#ffcccc"
        elif prediction < med_th:
            tag = '<span class="tag-medium">MEDIUM YIELD</span>'
            color = "#fff4cc"
        else:
            tag = '<span class="tag-high">HIGH YIELD</span>'
            color = "#d1ffcc"

    except Exception as e:
        st.error(f"⚠️ Error occurred: {e}")
        tag = '<span class="tag-low">LOW YIELD</span>'
        color = "#ffcccc"
        prediction = prediction / 2

    # -------- Prediction Card ----------
    st.markdown(
        f"""
        <div class="prediction-card" style="background:{color};">
            <h2>🌱 Predicted Yield: <b>{prediction:.2f} hg/ha</b></h2>
            <h3>{tag}</h3>
            <p>Based on climate, soil conditions, and farming inputs.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ------------------------------------------------------
    # Show Risk Warnings
    # ------------------------------------------------------
    if warnings:
        st.subheader("⚠️ Environmental Risk Analysis")
        for w in warnings:
            st.warning(w)
    else:
        st.success("🌤 Conditions look stable — no major risk detected.")

    # ------------------------------------------------------
    # Detailed Explanation
    # ------------------------------------------------------
    st.subheader("📘 Detailed Explanation of This Prediction")

    # ⬇ Temperature explanation
    st.markdown(f"### 🔥 Temperature Impact ({temperature}°C)")
    if temperature > 42:
        st.write("➡️ Extreme heat causes crop **burning and sterility**.")
    elif temperature > 35:
        st.write("➡️ Heat stress reduces **grain filling**.")
    else:
        st.write("➡️ Temperature is **optimal** for plant growth.")

    # ⬇ Rainfall explanation
    st.markdown(f"### 💧 Rainfall Impact ({rainfall} mm)")
    if rainfall > 2500:
        st.write("➡️ Flash floods and waterlogging cause **root death**.")
    elif rainfall < 300:
        st.write("➡️ Very low rainfall causes **drought stress**.")
    else:
        st.write("➡️ Rainfall is **sufficient**.")

    # ⬇ Pesticide explanation
    st.markdown(f"### 🧪 Pesticide Impact ({pesticides} tonnes)")
    if pesticides > 300:
        st.write("➡️ Toxicity risk → **soil microbes decline dramatically**.")
    elif pesticides < 1:
        st.write("➡️ Reduced protection → **pest infestation likely**.")
    else:
        st.write("➡️ Pesticide level is **balanced**.")

    # ------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------
    st.subheader("💡 Farming Recommendations")

    if prediction < 2000:
        st.warning("""
        **Low expected yield**. Suggestions:
        - Improve irrigation  
        - Use organic compost  
        - Adopt drought-resistant seed varieties  
        - Avoid excessive pesticide usage  
        """)
    elif prediction < 5000:
        st.info("""
        **Moderate yield**. Try:
        - Drip irrigation  
        - Hybrid crop varieties  
        - Soil nutrient monitoring  
        """)
    else:
        st.success("""
        **High yield expected!**
        - Maintain current practices  
        - Practice crop rotation  
        - Balanced fertilizer management  
        """)

    # ------------------------------------------------------
    # Input Summary
    # ------------------------------------------------------
    with st.expander("📁 Input Summary"):
        st.json({
            "Area": area,
            "Crop": item,
            "Rainfall (mm)": rainfall,
            "Pesticides (tonnes)": pesticides,
            "Temperature (°C)": temperature,
            "Risk factor applied": round(risk_factor, 3)
        })

else:
    st.info("Use the sidebar to enter values and click **Predict Yield**.")

st.divider()
st.caption("🌾 Built with Machine Learning + Agricultural Intelligence")