import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO
import datetime


# Page configuration
st.set_page_config(page_title="Water Treatment Prediction", layout="centered")

# Centered TTU Logo and Header Info
st.image("ttu_logo.png", width=800)
st.markdown("""
    <div style='text-align: center;'>
        <h3 style='color: green;'>Tafila Technical University</h3>
        <h4 style='color: green;'>Natural Resources and Chemical Engineering Department</h4>
        <p><strong>Bachelor's Degree Project</strong></p>
    </div>
""", unsafe_allow_html=True)

# Project Info Box
st.markdown("""
    <div style='border: 2px solid #ddd; padding: 15px; border-radius: 10px; margin-top: 10px;'>
        <h2 style='text-align: center;'>Modeling Coagulation–Flocculation with Artificial Neural Networks</h2>
        <h4 style='text-align: center;'>Operation Parameters Prediction</h4>
        <br>
        <p><strong>Students:</strong><br>
        Shahad Mohammed Abushamma<br>
        Rahaf Ramzi Al -shakh Qasem<br>
        Duaa Musa Al-Khalafat</p>
        <p><strong>Supervisor:</strong> Dr. Ashraf Alsafasfeh</p>
    </div>
""", unsafe_allow_html=True)

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

# Streamlit App Header
st.markdown("""
    <h1 style='text-align: center; color: #0072B2;'>Water Treatment Prediction App by NRCE's Students</h1>
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
        predicted_output[predicted_output < 0] = 0

        final_outputs = predicted_output[0]
        quality_vars = output_vars[:7]
        process_vars = output_vars[10:]

        # Operational Parameters
        st.subheader("⚙️ Predicted Operational Parameters")
        st.info("To ensure safe water quality, predicted operational parameters should be considered minimum values ±6%.")
        units = [
            "mg/L", "mg/L", "rpm", "min", "min", "min"
        ]
        data_process = []
        for i, var in enumerate(process_vars):
            val = final_outputs[i + 10]
            data_process.append([var.replace('_', ' '), f"{val:.2f}", units[i]])
        df_process = pd.DataFrame(data_process, columns=["Parameter", "Predicted Value", "Unit"])
        st.table(df_process)

        # Treated Water Quality
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

        data = []
        safe = True
        for i, var in enumerate(quality_vars):
            limit = limits[var]
            val = max(0, final_outputs[i])
            if val > limit:
                val = limit * 1.005  # Cap at +0.5% of limit
                status = "❌"
                safe = False
            else:
                status = "✅"
            data.append([var.replace('_', ' '), f"{val:.2f}", f"≤ {limit}", status])

        df_display = pd.DataFrame(data, columns=["Parameter", "Predicted Value", "Limit", "Status"])
        st.table(df_display)

        # Optional Plot
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
            st.error("❌ Result: Water is NOT safe for reuse or discharge. Unless if the predicted values are capped at +1% of their respective limits")
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
            st.info("Try adjusting one parameter at a time, as recall, the predicted operational parameters should be considered minimum values ±6%.")
  # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Water Quality Assessment Report", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Date: {datetime.date.today()}", ln=True, align="C")

    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Input Parameters:", ln=True)

    # Display the input parameters in the report
    user_inputs = {
    "Temperature (°C)": st.number_input("Enter temperature (°C):", value=25),
    "pH": st.number_input("Enter pH level:", value=7.0),
    "Turbidity (NTU)": st.number_input("Enter turbidity (NTU):", value=1.0),
}
    for var, value in user_inputs.items():
        pdf.cell(200, 10, txt=f"{var}: {value}", ln=True)

    # Save and allow file download
    pdf_file = "water_quality_report.pdf"
    pdf.output(pdf_file)

    # Enable the user to download the generated report
    with open(pdf_file, "rb") as file:
        st.download_button("Download Report", file, file_name=pdf_file)
