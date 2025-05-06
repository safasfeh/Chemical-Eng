import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib  # For loading models if not saved in H5 format

# Streamlit UI for uploading files
st.title("Water Quality Prediction")

# Upload model and scaler files
model_file = st.file_uploader("Upload Model File (.h5 or .pkl)", type=["h5", "pkl"])
scaler_file_X = st.file_uploader("Upload Scaler (X) File (.pkl)", type=["pkl"])
scaler_file_y = st.file_uploader("Upload Scaler (y) File (.pkl)", type=["pkl"])

if model_file is not None and scaler_file_X is not None and scaler_file_y is not None:
    # Load the model
    if model_file.name.endswith('.h5'):
        model = load_model(model_file)
    elif model_file.name.endswith('.pkl'):
        model = joblib.load(model_file)

    # Load the scalers
    scaler_X = joblib.load(scaler_file_X)
    scaler_y = joblib.load(scaler_file_y)

    # Function to handle input prediction
    def predict_quality(new_input):
        # Scale the new input data using the scaler
        new_input_scaled = scaler_X.transform(new_input)
        
        # Predict the output
        final_outputs_scaled = model.predict(new_input_scaled)
        
        # Inverse transform to get the predicted values in original scale
        final_outputs = scaler_y.inverse_transform(final_outputs_scaled)
        
        # Ensure no predicted values are less than 0
        final_outputs = np.maximum(final_outputs, 0)
        
        return final_outputs[0]

    # Streamlit UI for input parameters
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=1.0, step=0.1)
    temp = st.number_input("Temperature (°C)", min_value=0.0, value=25.0, step=0.1)
    fe = st.number_input("Fe (mg/L)", min_value=0.0, value=0.5, step=0.1)
    mn = st.number_input("Mn (mg/L)", min_value=0.0, value=0.1, step=0.01)
    cu = st.number_input("Cu (mg/L)", min_value=0.0, value=0.05, step=0.01)
    zn = st.number_input("Zn (mg/L)", min_value=0.0, value=0.02, step=0.01)
    ss = st.number_input("Suspended Solids (mg/L)", min_value=0.0, value=5.0, step=0.1)
    tds = st.number_input("TDS (mg/L)", min_value=0.0, value=100.0, step=1)

    # Create a new input array with the parameters
    new_input = np.array([[ph, turbidity, temp, fe, mn, cu, zn, ss, tds]])

    # Predict water quality based on input
    final_outputs = predict_quality(new_input)

    # Display the predicted water quality parameters
    st.subheader("Predicted Treated Water Quality Parameters")
    predicted_df = pd.DataFrame({
        'Parameter': ['pH', 'Turbidity (NTU)', 'Temperature (°C)', 'Fe (mg/L)', 'Mn (mg/L)', 'Cu (mg/L)', 
                    'Zn (mg/L)', 'Suspended Solids (mg/L)', 'TDS (mg/L)'],
        'Predicted Value': final_outputs
    })
    st.table(predicted_df)

    # Prepare data for plotting comparison between raw and predicted water quality
    raw_values = [ph, turbidity, temp, fe, mn, cu, zn, ss, tds]
    predicted_values = final_outputs[:9]  # Match the length of raw inputs

    # Create a bar chart for the comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = np.arange(len(raw_values))

    ax.barh(index, raw_values, bar_width, label='Raw Water Quality Parameters', color='lightgray')
    ax.barh(index + bar_width, predicted_values, bar_width, label='Predicted Treated Water Quality', color='skyblue')

    ax.set_xlabel('Values')
    ax.set_ylabel('Parameters')
    ax.set_title('Comparison between Raw Water Quality and Predicted Treated Water Quality')
    ax.set_yticks(index + bar_width / 2)
    ax.set_yticklabels(['pH', 'Turbidity (NTU)', 'Temperature (°C)', 'Fe (mg/L)', 'Mn (mg/L)', 'Cu (mg/L)', 
                        'Zn (mg/L)', 'Suspended Solids (mg/L)', 'TDS (mg/L)'])

    ax.legend()
    st.pyplot(fig)

    # Show predicted operational parameters with ±5% error
    operation_limits = {
        'Coagulant_dose_mg_L': 20.0,
        'Flocculant_dose_mg_L': 5.0,
        'Mixing_speed_rpm': 100.0,
        'Rapid_mix_time_min': 5.0,
        'Slow_mix_time_min': 15.0,
        'Settling_time_min': 30.0
    }

    operation_table_data = []
    for var, limit in operation_limits.items():
        min_value = limit - (limit * 0.05)
        max_value = limit + (limit * 0.05)
        operation_table_data.append([var, f"{min_value:.2f} - {max_value:.2f}"])

    st.subheader("⚙️ Predicted Operational Parameters (with ±5% error)")
    operation_df = pd.DataFrame(operation_table_data, columns=["Parameter", "Required Range (±5%)"])
    st.table(operation_df)

    # Optimization function: adjust one parameter at a time
    def optimize_parameter(parameter_index, step_size=0.1):
        optimized_inputs = np.copy(new_input)
        
        # Increment the chosen parameter by step_size and re-run prediction
        optimized_inputs[0, parameter_index] += step_size
        optimized_input_scaled = scaler_X.transform(optimized_inputs)
        optimized_output_scaled = model.predict(optimized_input_scaled)
        optimized_output = scaler_y.inverse_transform(optimized_output_scaled)
        
        return optimized_output[0]

    # Example of optimizing a parameter (adjust pH by 0.1 and re-run prediction)
    optimized_values = optimize_parameter(0, step_size=0.1)  # Adjusting pH (index 0)
    st.subheader("Optimized Prediction (after adjusting pH)")
    optimized_df = pd.DataFrame({
        'Parameter': ['pH', 'Turbidity (NTU)', 'Temperature (°C)', 'Fe (mg/L)', 'Mn (mg/L)', 'Cu (mg/L)', 
                    'Zn (mg/L)', 'Suspended Solids (mg/L)', 'TDS (mg/L)'],
        'Optimized Value': optimized_values
    })
    st.table(optimized_df)

else:
    st.warning("Please upload the required files (model and scalers).")
