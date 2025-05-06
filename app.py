import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load data and model
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
model = load_model("model.h5")
df = pd.read_excel("simulated_mine_water_treatment_data.xlsx")

# Define variables
input_vars = [
    'pH_raw', 'Turbidity_raw_NTU', 'Temperature_C', 'Fe_initial_mg_L', 'Mn_initial_mg_L',
    'Cu_initial_mg_L', 'Zn_initial_mg_L', 'Suspended_solids_mg_L', 'TDS_mg_L'
]
output_vars = [
    'Turbidity_final_NTU', 'Fe_final_mg_L', 'Mn_final_mg_L', 'Cu_final_mg_L', 'Zn_final_mg_L',
    'Suspended_solids_final_mg_L', 'TDS_final_mg_L',
    'Turbidity_removal_%', 'Suspended_solids_removal_%', 'TDS_removal_%',
    'Coagulant_dose_mg_L', 'Flocculant_dose_mg_L', 'Mixing_speed_rpm',
    'Rapid_mix_time_min', 'Slow_mix_time_min', 'Settling_time_min'
]

X = df[input_vars].values
y = df[output_vars].values
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Streamlit app
st.title("Mine Water Treatment Prediction System")

st.header("📥 Enter the raw water quality parameters below:")
with st.form("input_form"):
    pH_raw = st.number_input("pH (0–14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    turbidity = st.number_input("Turbidity (NTU)", value=50.0)
    temp = st.number_input("Temperature (°C)", value=25.0)
    fe = st.number_input("Initial Fe (mg/L)", value=1.0)
    mn = st.number_input("Initial Mn (mg/L)", value=0.2)
    cu = st.number_input("Initial Cu (mg/L)", value=0.05)
    zn = st.number_input("Initial Zn (mg/L)", value=0.1)
    ss = st.number_input("Suspended Solids (mg/L)", value=150.0)
    tds = st.number_input("TDS (mg/L)", value=900.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    new_input = np.array([[pH_raw, turbidity, temp, fe, mn, cu, zn, ss, tds]])
    new_input_scaled = scaler_X.transform(new_input)
    predicted_scaled = model.predict(new_input_scaled)
    predicted = scaler_y.inverse_transform(predicted_scaled)
    predicted[predicted < 0] = 0  # Clamp negatives to 0

    final_outputs = dict(zip(output_vars, predicted[0]))
    quality_outputs = {k: final_outputs[k] for k in output_vars[:7]}
    operational_outputs = {k: final_outputs[k] for k in output_vars[10:]}

    st.subheader("⚙️ Predicted Operational Parameters")
    st.markdown("These are the **minimum required values** with ±5% variation to achieve the predicted treated water quality.")
    for k, v in operational_outputs.items():
        st.write(f"{k}: {v:.2f}")

    st.subheader("🧪 Predicted Treated Water Quality")
    quality_limits = {
        'Turbidity_final_NTU': 5.0,
        'Fe_final_mg_L': 0.3,
        'Mn_final_mg_L': 0.1,
        'Cu_final_mg_L': 1.0,
        'Zn_final_mg_L': 5.0,
        'Suspended_solids_final_mg_L': 50.0,
        'TDS_final_mg_L': 1000.0
    }
    
    # Display table with signs
    results_table = pd.DataFrame([
        [k, f"{quality_outputs[k]:.2f}", f"≤ {quality_limits[k]}", "✅" if quality_outputs[k] <= quality_limits[k] else "❌"]
        for k in quality_limits
    ], columns=["Parameter", "Predicted Value", "Max Acceptable Limit", "Status"])
    st.table(results_table)

    # Only show safe result message
    if all(quality_outputs[k] <= quality_limits[k] for k in quality_limits):
        st.success("✅ Result: Water is safe for reuse or discharge.")

    # Charts
    st.subheader("📊 Comparison Charts")
    fig, ax = plt.subplots(figsize=(10, 5))
    raw_vals = [pH_raw, turbidity, temp, fe, mn, cu, zn, ss, tds]
    treated_vals = [quality_outputs[k] for k in quality_limits]
    labels = list(quality_limits.keys())
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, treated_vals, width, label='Treated Water')
    ax.bar(x + width/2, [quality_limits[k] for k in quality_limits], width, label='Standard Limit')
    ax.set_ylabel('Concentration / NTU')
    ax.set_title('Treated Water Quality vs. Standard Limits')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(x - width/2, raw_vals[0:7], width, label='Raw Water')
    ax2.bar(x + width/2, treated_vals, width, label='Treated Water')
    ax2.set_ylabel('Concentration / NTU')
    ax2.set_title('Raw vs. Treated Water Quality')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    st.pyplot(fig2)
