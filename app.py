import streamlit as st
import numpy as np
from model import load_and_train_model

# ================================
# Load model each time (NO .pkl files)
# ================================
model, le_area, le_item, scaler = load_and_train_model()

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
# Sidebar Inputs (unchanged)
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
# SAME AGRICULTURAL WARNINGS + RISK LOGIC
# ------------------------------------------------------
warnings = []
risk_factor = 0.0

# Rainfall logic
if rainfall > 2500:
    warnings.append("💧 **Dangerously high rainfall** → flooding.")
    risk_factor += 0.40
elif rainfall > 1800:
    warnings.append("💧 **Very high rainfall** → waterlogging.")
    risk_factor += 0.25
elif rainfall < 300:
    warnings.append("💧 **Severe drought risk**.")
    risk_factor += 0.30
elif rainfall < 600:
    warnings.append("💧 **Low rainfall** → irrigation needed.")
    risk_factor += 0.15

# Pesticides logic
if pesticides > 300:
    warnings.append("🧪 **Toxic pesticide levels**.")
    risk_factor += 0.35
elif pesticides > 150:
    warnings.append("🧪 **High pesticide usage**.")
    risk_factor += 0.20
elif pesticides < 1:
    warnings.append("🧪 **Too low pesticide usage**.")
    risk_factor += 0.12

# Temperature logic
if temperature > 42:
    warnings.append("🔥 **Extreme heat (>42°C)**.")
    risk_factor += 0.45
elif temperature > 35:
    warnings.append("🔥 **High heat stress**.")
    risk_factor += 0.25
elif temperature < 12:
    warnings.append("❄️ **Cold stress**.")
    risk_factor += 0.20


# ------------------------------------------------------
# Prepare Input (same)
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
    prediction *= (1 - risk_factor)
    prediction = max(prediction, 100)

    # ---------------- Tag logic exactly same ----------------
    try:
        if risk_factor > 0.60:
            low_th = 10000
            med_th = 20000
        elif risk_factor > 0.30:
            low_th = 7000
            med_th = 14000
        else:
            low_th = 2000
            med_th = 5000

        if prediction < low_th:
            tag = '<span class="tag-low">LOW YIELD</span>'
            color = "#ffcccc"
        elif prediction < med_th:
            tag = '<span class="tag-medium">MEDIUM YIELD</span>'
            color = "#fff4cc"
        else:
            tag = '<span class="tag-high">HIGH YIELD</span>'
            color = "#d1ffcc"

    except:
        tag = '<span class="tag-low">LOW YIELD</span>'
        color = "#ffcccc"

    # Prediction card
    st.markdown(
        f"""
        <div class="prediction-card" style="background:{color};">
            <h2>🌱 Predicted Yield: <b>{prediction:.2f} hg/ha</b></h2>
            <h3>{tag}</h3>
            <p>Based on climate and farming inputs.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Risk warnings
    if warnings:
        st.subheader("⚠️ Environmental Risk Analysis")
        for w in warnings:
            st.warning(w)
    else:
        st.success("🌤 Conditions look stable.")

    # Explanations (unchanged)
    st.subheader("📘 Detailed Explanation")
    st.markdown(f"### 🔥 Temperature Impact ({temperature}°C)")
    st.markdown(f"### 💧 Rainfall Impact ({rainfall} mm)")
    st.markdown(f"### 🧪 Pesticide Impact ({pesticides} tonnes)")

    # Recommendations (unchanged)
    st.subheader("💡 Farming Recommendations")
    if prediction < 2000:
        st.warning("Low yield suggestions...")
    elif prediction < 5000:
        st.info("Moderate yield tips...")
    else:
        st.success("High yield — maintain practices!")

    with st.expander("📁 Input Summary"):
        st.json({
            "Area": area,
            "Crop": item,
            "Rainfall": rainfall,
            "Pesticides": pesticides,
            "Temperature": temperature,
            "Risk factor": round(risk_factor, 3),
        })

else:
    st.info("Use the sidebar to enter values and click Predict.")

st.divider()
st.caption("🌾 Built with Machine Learning")
