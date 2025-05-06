import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("simulated_mine_water_treatment_data.xlsx")

# Define variables
input_vars = ['pH_raw', 'Turbidity_raw_NTU', 'Temperature_C', 'Fe_initial_mg_L',
              'Mn_initial_mg_L', 'Cu_initial_mg_L', 'Zn_initial_mg_L',
              'Suspended_solids_mg_L', 'TDS_mg_L']
output_vars = ['Turbidity_final_NTU', 'Fe_final_mg_L', 'Mn_final_mg_L', 'Cu_final_mg_L',
               'Zn_final_mg_L', 'Suspended_solids_final_mg_L', 'TDS_final_mg_L',
               'Turbidity_removal_%', 'Suspended_solids_removal_%', 'TDS_removal_%',
               'Coagulant_dose_mg_L', 'Flocculant_dose_mg_L', 'Mixing_speed_rpm',
               'Rapid_mix_time_min', 'Slow_mix_time_min', 'Settling_time_min']

# Data preparation
X = df[input_vars].values
y = df[output_vars].values
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Model training
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='linear')
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.1, epochs=150, batch_size=16, verbose=0)

# Streamlit App Interface
st.title("💧 Mine Water Treatment Prediction App")
st.subheader("📥 Enter the raw water quality parameters below:")

# Input form
with st.form("input_form"):
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.5)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=45.0)
    temp = st.number_input("Temperature (°C)", value=25.0)
    fe = st.number_input("Fe (mg/L)", value=1.2)
    mn = st.number_input("Mn (mg/L)", value=0.3)
    cu = st.number_input("Cu (mg/L)", value=0.05)
    zn = st.number_input("Zn (mg/L)", value=0.1)
    ss = st.number_input("Suspended Solids (mg/L)", value=150.0)
    tds = st.number_input("TDS (mg/L)", value=1000.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    new_input = np.array([[ph, turbidity, temp, fe, mn, cu, zn, ss, tds]])
    new_input_scaled = scaler_X.transform(new_input)
    predicted_output_scaled = model.predict(new_input_scaled)
    predicted_output = scaler_y.inverse_transform(predicted_output_scaled)
    final_outputs = np.maximum(predicted_output[0], 0)  # Clip negatives to zero

    # Variables split
    quality_vars = output_vars[:7]
    process_vars = output_vars[10:]

    if safe:
      # ✅ Water is safe
      st.success("✅ Result: Water is safe for reuse or discharge.")
  
      # ⚙️ Predicted Operational Parameters (only shown if water is safe)
      st.subheader("⚙️ Predicted Operational Parameters")
      st.markdown("_These values represent recommended dosages and process times (±5% tolerance)._")
  
      # Units for each operational parameter
      op_units = {
          'Coagulant_dose_mg_L': 'mg/L',
          'Flocculant_dose_mg_L': 'mg/L',
          'Mixing_speed_rpm': 'rpm',
          'Rapid_mix_time_min': 'min',
          'Slow_mix_time_min': 'min',
          'Settling_time_min': 'min'
      }
  
      # Build table
      op_data = []
      for i, var in enumerate(process_vars, start=10):
          name = var.replace('_', ' ')
          value = final_outputs[i]
          unit = op_units.get(var, "")
          op_data.append([name, f"{value:.2f}", unit])
  
      df_ops = pd.DataFrame(op_data, columns=["Parameter", "Value", "Unit"])
      st.table(df_ops)


    # 🧪 Treated Water Quality Results
    st.subheader("🧪 Predicted Treated Water Quality")

    limits = {
        'Turbidity_final_NTU': 5.0,
        'Fe_final_mg_L': 0.3,
        'Mn_final_mg_L': 0.1,
        'Cu_final_mg_L': 1.0,
        'Zn_final_mg_L': 5.0,
        'Suspended_solids_final_mg_L': 50.0,
        'TDS_final_mg_L': 1000.0
    }

    safe = True
    data = []
    for i, var in enumerate(quality_vars):
        val = final_outputs[i]
        limit = limits[var]
        status = "✅" if val <= limit else "❌"
        if status == "❌":
            safe = False
        data.append([var.replace('_', ' '), f"{val:.2f}", f"≤ {limit}", status])

    df_display = pd.DataFrame(data, columns=["Parameter", "Predicted Value", "Limit", "Status"])
    st.table(df_display)

    # ✅ Show only if safe
    if safe:
        st.success("✅ Result: Water is safe for reuse or discharge.")

        # Chart 1: Raw vs Treated
        st.subheader("📊 Raw vs Predicted Treated Values")
        raw_vals = [ph, turbidity, temp, fe, mn, cu, zn, ss, tds]
        treated_vals = final_outputs[:7]
        labels = ['pH', 'Turbidity', 'Fe', 'Mn', 'Cu', 'Zn', 'SS', 'TDS']

        fig1, ax1 = plt.subplots()
        x = np.arange(len(labels))
        ax1.bar(x - 0.2, raw_vals[:len(labels)], width=0.4, label='Raw Input', color='orange')
        ax1.bar(x + 0.2, treated_vals, width=0.4, label='Treated Output', color='green')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45)
        ax1.set_ylabel("mg/L or NTU")
        ax1.set_title("Raw vs Treated Water Quality")
        ax1.legend()
        st.pyplot(fig1)

        # Chart 2: Treated vs Limits
        st.subheader("📊 Treated Quality vs Acceptable Limits")
        limits_vals = [limits[var] for var in limits]
        treated_vals = [float(row[1]) for row in data]
        labels = [row[0] for row in data]

        fig2, ax2 = plt.subplots()
        x = np.arange(len(labels))
        ax2.bar(x - 0.2, limits_vals, width=0.4, label='Limit', color='gray')
        ax2.bar(x + 0.2, treated_vals, width=0.4, label='Treated', color='blue')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45)
        ax2.set_ylabel("mg/L or NTU")
        ax2.set_title("Treated Water Quality vs Standards")
        ax2.legend()
        st.pyplot(fig2)
