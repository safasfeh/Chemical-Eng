import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import os

st.set_page_config(page_title="Mine Water Treatment Prediction", layout="wide")

@st.cache_data
def load_data():
    return pd.read_excel("simulated_mine_water_treatment_data.xlsx")

@st.cache_resource
def train_model(df):
    input_vars = [
        'pH_raw', 'Turbidity_raw_NTU', 'Temperature_C', 'Fe_initial_mg_L',
        'Mn_initial_mg_L', 'Cu_initial_mg_L', 'Zn_initial_mg_L',
        'Suspended_solids_mg_L', 'TDS_mg_L'
    ]
    output_vars = [
        'Turbidity_final_NTU', 'Fe_final_mg_L', 'Mn_final_mg_L', 'Cu_final_mg_L',
        'Zn_final_mg_L', 'Suspended_solids_final_mg_L', 'TDS_final_mg_L',
        'Turbidity_removal_%', 'Suspended_solids_removal_%', 'TDS_removal_%',
        'Coagulant_dose_mg_L', 'Flocculant_dose_mg_L', 'Mixing_speed_rpm',
        'Rapid_mix_time_min', 'Slow_mix_time_min', 'Settling_time_min'
    ]

    X = df[input_vars].values
    y = df[output_vars].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(64, activation='relu'),
        Dense(y_train.shape[1], activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, validation_split=0.1, epochs=150, batch_size=16, verbose=0)

    return model, scaler_X, scaler_y, output_vars

# Load and train
df = load_data()
model, scaler_X, scaler_y, output_vars = train_model(df)

st.title("💧 Mine Water Treatment Prediction and Reuse Assessment")

st.markdown("Enter the raw water quality parameters below:")

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        pH = st.number_input("pH_raw", value=7.5)
        turbidity = st.number_input("Turbidity_raw_NTU", value=45.0)
        temp = st.number_input("Temperature_C", value=25.0)

    with col2:
        fe = st.number_input("Fe_initial_mg_L", value=1.2)
        mn = st.number_input("Mn_initial_mg_L", value=0.3)
        cu = st.number_input("Cu_initial_mg_L", value=0.05)

    with col3:
        zn = st.number_input("Zn_initial_mg_L", value=0.1)
        ss = st.number_input("Suspended_solids_mg_L", value=150.0)
        tds = st.number_input("TDS_mg_L", value=1000.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_array = np.array([[pH, turbidity, temp, fe, mn, cu, zn, ss, tds]])
    input_scaled = scaler_X.transform(input_array)
    prediction_scaled = model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled)[0]

    final_quality = output_vars[:7]
    operational_params = output_vars[7:]

    limits = {
        'Turbidity_final_NTU': (0.0, 5.0),
        'Fe_final_mg_L': (0.0, 0.3),
        'Mn_final_mg_L': (0.0, 0.1),
        'Cu_final_mg_L': (0.0, 1.0),
        'Zn_final_mg_L': (0.0, 5.0),
        'Suspended_solids_final_mg_L': (0.0, 50.0),
        'TDS_final_mg_L': (0.0, 1000.0)
    }

    st.subheader("🧪 Predicted Treated Water Quality:")
    reuse_safe = True
    for i, var in enumerate(final_quality):
        val = prediction[i]
        in_range = limits[var][0] <= val <= limits[var][1]
        status = "✅ Within limit" if in_range else "❌ Out of limit"
        if not in_range:
            reuse_safe = False
        st.write(f"{var}: {val:.3f} ({status})")

    st.subheader("⚙️ Predicted Operational Parameters:")
    for i, var in enumerate(operational_params, start=7):
        st.write(f"{var}: {prediction[i]:.3f}")

    if reuse_safe:
        st.success("✅ Result: Water is safe for reuse or discharge.")
    else:
        st.error("❌ Result: Water is NOT safe for reuse or discharge.")
