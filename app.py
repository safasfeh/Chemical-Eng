import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel("simulated_mine_water_treatment_data.xlsx")

# Define variables
input_vars = [
    'pH_raw', 'Turbidity_raw_NTU', 'Temperature_C', 'Fe_initial_mg_L', 'Mn_initial_mg_L', 'Cu_initial_mg_L',
    'Zn_initial_mg_L', 'Suspended_solids_mg_L', 'TDS_mg_L'
]
output_vars = [
    'Turbidity_final_NTU', 'Fe_final_mg_L', 'Mn_final_mg_L', 'Cu_final_mg_L',
    'Zn_final_mg_L', 'Suspended_solids_final_mg_L', 'TDS_final_mg_L',
    'Turbidity_removal_%', 'Suspended_solids_removal_%', 'TDS_removal_%', 'Coagulant_dose_mg_L',
    'Flocculant_dose_mg_L', 'Mixing_speed_rpm',
    'Rapid_mix_time_min', 'Slow_mix_time_min', 'Settling_time_min'
]

# Prepare data
X = df[input_vars].values
y = df[output_vars].values
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, validation_split=0.1, epochs=150, batch_size=16, verbose=0)

# Streamlit App
st.image("ttu_logo.png", width=800)
st.markdown("""
<div style='text-align: center;'>
    <h3 style='color: green;'>Tafila Technical University<br>Natural Resources and Chemical Engineering Department</h3>
    <h4>Bachelor's Degree Project</h4>
</div>
<div style='border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; margin: 10px;'>
    <h2 style='text-align: center;'>Modeling Coagulation–Flocculation with Artificial Neural Networks</h2>
    <h3 style='text-align: center;'>Operation Parameters Prediction</h3>
    <p><strong>Students:</strong><br>
    Shahad Mohammed Abushamma<br>
    Rahaf Ramzi Al -shakh Qasem<br>
    Duaa Musa Al-Khalafat</p>
    <p><strong>Supervisor:</strong> Dr. Ashraf Alsafasfeh</p>
</div>
""", unsafe_allow_html=True)

st.subheader("📥 Enter the raw water quality parameters below:")

def is_valid_ph(ph):
    return 0 <= ph <= 14

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
    if not is_valid_ph(ph):
        st.error("❌ Invalid pH: Please enter a value between 0 and 14.")
    else:
        new_input = np.array([[ph, turbidity, temp, fe, mn, cu, zn, ss, tds]])
        new_input_scaled = scaler_X.transform(new_input)
        predicted_output_scaled = model.predict(new_input_scaled)
        predicted_output = scaler_y.inverse_transform(predicted_output_scaled)
        final_outputs = predicted_output[0]

        process_vars = output_vars[10:]
        operational_data = []
        for i, var in enumerate(process_vars, start=10):
            value = final_outputs[i]
            if var.endswith("_mg_L"):
                unit = "mg/L"
            elif var.endswith("_rpm"):
                unit = "rpm"
            elif var.endswith("_min"):
                unit = "min"
            else:
                unit = ""
            operational_data.append([var.replace('_', ' '), f"{value:.2f}", unit])

        operational_df = pd.DataFrame(operational_data, columns=["Parameter", "Predicted Value", "Unit"])
        st.subheader("⚙️ Predicted Operational Parameters")
        st.info("To achieve good water treatment results, operational parameters should be at minimum these predicted values with ±2% margin.")
        st.table(operational_df)

        quality_vars = output_vars[:7]
        limits = {
            'Turbidity_final_NTU': 5.0,
            'Fe_final_mg_L': 0.3,
            'Mn_final_mg_L': 0.1,
            'Cu_final_mg_L': 1.0,
            'Zn_final_mg_L': 5.0,
            'Suspended_solids_final_mg_L': 50.0,
            'TDS_final_mg_L': 1000.0
        }

        st.subheader("🧪 Predicted Treated Water Quality")
        data = []
        safe = True
        for i, var in enumerate(quality_vars):
            val = final_outputs[i]
            limit = limits[var]
            status = "✅" if val <= limit else "❌"
            if status == "❌":
                safe = False
            data.append([var.replace('_', ' '), f"{val:.2f}", f"≤ {limit}", status])

        df_display = pd.DataFrame(data, columns=["Parameter", "Predicted Value", "Limit", "Status"])
        st.table(df_display)

        fig, ax = plt.subplots()
        values = [float(row[1]) for row in data]
        limits_list = [float(row[2].split()[-1]) for row in data]
        labels = [row[0] for row in data]
        ax.barh(labels, limits_list, color='lightgray', label="Limit")
        ax.barh(labels, values, color='skyblue', alpha=0.8, label="Predicted")
        ax.legend()
        st.pyplot(fig)

        if safe:
            st.success("✅ Result: Water is safe for reuse or discharge.")
        else:
            st.error("❌ Result: Water is NOT safe for reuse or discharge.")
            st.subheader("🔁 Suggested Operational Adjustments")
            st.markdown(
                "- **Increase Coagulant dose**: Improves the removal of suspended solids and metals. `This can help reduce turbidity, Fe, Mn.`"
            )
            st.markdown(
                "- **Increase Flocculant dose**: Enhances floc formation for better sedimentation. `This supports lowering of TDS and fine solids.`"
            )
            st.markdown(
                "- **Increase Settling time**: Allows more particles to settle out of solution. `This improves clarity and reduces final turbidity.`"
            )
            st.markdown(
                "- **Adjust Mixing Speed/Time**: Better mixing can improve contact efficiency of chemicals. `Slower mixing during flocculation can improve settling behavior.`"
            )
            st.info("Try adjusting one parameter at a time and re-run the prediction.")
