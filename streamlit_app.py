import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load("tensflow.joblib")  # Adjust the path if needed

# Encode mappings as dictionaries
location_mapping = {
    'Ahmedabad': 0, 'Bangalore': 1, 'Chennai': 2, 'Coimbatore': 3, 'Delhi': 4,
    'Hyderabad': 5, 'Jaipur': 6, 'Kochi': 7, 'Kolkata': 8, 'Mumbai': 9, 'Pune': 10
}

fuel_type_mapping = {
    'CNG': 0, 'Diesel': 1, 'Electric': 2, 'LPG': 3, 'Petrol': 4
}

transmission_mapping = {
    'Automatic': 0, 'Manual': 1
}

owner_type_mapping = {
    'First': 0, 'Fourth & Above': 1, 'Second': 2, 'Third': 3
}

brand_mapping = {
    'Ambassador': 0, 'Audi': 1, 'BMW': 2, 'Bentley': 3, 'Chevrolet': 4,
    'Datsun': 5, 'Fiat': 6, 'Force': 7, 'Ford': 8, 'Hindustan': 9, 'Honda': 10,
    'Hyundai': 11, 'ISUZU': 12, 'Isuzu': 13, 'Jaguar': 14, 'Jeep': 15,
    'Lamborghini': 16, 'Land': 17, 'Mahindra': 18, 'Maruti': 19, 'Mercedes-Benz': 20,
    'Mini': 21, 'Mitsubishi': 22, 'Nissan': 23, 'OpelCorsa': 24, 'Porsche': 25,
    'Renault': 26, 'Skoda': 27, 'Smart': 28, 'Tata': 29, 'Toyota': 30,
    'Volkswagen': 31, 'Volvo': 32
}

# Streamlit UI Setup
st.set_page_config(page_title="Car Sale Price Predictor", layout="wide")
st.title("🚗 **Car Sale Price Prediction**")

# Input fields for each feature
location = st.selectbox('Select Location', options=list(location_mapping.keys()))
fuel_type = st.selectbox('Select Fuel Type', options=list(fuel_type_mapping.keys()))
transmission = st.selectbox('Select Transmission', options=list(transmission_mapping.keys()))
owner_type = st.selectbox('Select Owner Type', options=list(owner_type_mapping.keys()))
brand = st.selectbox('Select Brand', options=list(brand_mapping.keys()))

year = st.number_input("Enter Car Age (in years):", min_value=1, max_value=50, value=5, step=1)
km = st.number_input("Enter Kilometer Driven (in kms):", min_value=0, max_value=300000, value=50000, step=1000)
mileage = st.number_input("Enter Car Mileage (km/l):", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
cc = st.number_input("Enter Car Engine Capacity (CC):", min_value=500, max_value=8000, value=1500, step=50)
power = st.number_input("Enter Car Power (bhp):", min_value=0, max_value=1000, value=100, step=10)
seat = st.number_input("Enter Number of Seats:", min_value=2, max_value=8, value=5, step=1)

# Displaying a preview of the inputs
st.subheader("Entered Details:")
st.write(f"**Location:** {location}          |    **Fuel Type:** {fuel_type}    |    **Transmission:** {transmission}")
st.write(f"**Owner Type:** {owner_type}      |    **Brand:** {brand}            |    **Car Age:** {year} years")
st.write(f"**Kilometers Driven:** {km} km    |   **Mileage:** {mileage} km/l    |    **Engine Capacity:** {cc} CC")
st.write(f"**Power:** {power} bhp            |   **Seats:** {seat}")
st.write(f"**Seats:** {seat}")

# Function to process inputs and predict
def prepare_input_data():
    # Encoding the categorical features
    encoded_location = location_mapping[location]
    encoded_fuel_type = fuel_type_mapping[fuel_type]
    encoded_transmission = transmission_mapping[transmission]
    encoded_owner_type = owner_type_mapping[owner_type]
    encoded_brand = brand_mapping[brand]

    # Preparing the input data for prediction
    input_data = np.array([[encoded_location, year, km, encoded_fuel_type, encoded_transmission,
                            encoded_owner_type, mileage, cc, power, seat, encoded_brand]])
    return input_data

# Prediction button and display
if st.button('Predict Car Sale Value'):
    # Prepare input data
    input_data = prepare_input_data()

    # Make prediction using the model
    predicted_value = model.predict(input_data)

    # Check if predicted_value is a list or an array and ensure it is a scalar
    predicted_value = predicted_value[0] if isinstance(predicted_value, np.ndarray) else predicted_value

    # Ensure the predicted value is a numeric type (float or int)
    predicted_value = float(predicted_value)  # Convert to float, in case it's not already

    # Ensure the predicted value is non-negative by taking the absolute value
    predicted_value = abs(predicted_value)  # Remove negative sign if it exists

    # Show the predicted sale price
    st.subheader("Predicted Car Sale Value:")
    st.write(f"₹{predicted_value:,.2f}")  # Format the value as currency with commas and two decimal places

    # Additional user-friendly information
    st.write(f"This prediction is based on the details you provided and might vary with market conditions.")



# Add a footer section for additional information
st.markdown("---")
st.markdown("Made with ❤️ by Utkarsh Mishra")
st.markdown("Feel free to reach out for any feedback or suggestions.")
