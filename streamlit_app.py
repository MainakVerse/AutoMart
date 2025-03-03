import streamlit as st
import numpy as np
import joblib
import time
import random
from PIL import Image
import base64
import google.generativeai as genai

# Set page configuration with dark theme
st.set_page_config(
    page_title="AutoVault: Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for the techy theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    :root {
        --background-color: #121212;
        --text-color: #ff3300;
        --accent-color: #ff0000;
        --secondary-color: #ff6600;
    }
    
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: 'Share Tech Mono', monospace;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--accent-color) !important;
        font-family: 'Share Tech Mono', monospace;
    }
    
    .stButton button {
        background-color: var(--secondary-color);
        color: white;
        border: 1px solid var(--accent-color);
    }
    
    .stSelectbox > div > div {
        background-color: #1E1E1E;
        color: var(--text-color);
    }
    
    .stNumberInput > div > div > input {
        color: var(--text-color);
        background-color: #1E1E1E;
    }
    
    .stTextInput > div > div > input {
        color: var(--text-color);
        background-color: #1E1E1E;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E1E;
        border-radius: 4px 4px 0 0;
        border: 1px solid var(--secondary-color);
        border-bottom: none;
        color: var(--text-color);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--secondary-color);
        color: white;
    }
    
    div.stTabs > div > div:first-of-type {
        border-bottom: 1px solid var(--secondary-color);
    }
    
    .css-1dp5vir {
        background-image: linear-gradient(90deg, rgb(18, 18, 18), rgb(18, 18, 18));
    }
    
    .card {
        background-color: #1E1E1E;
        border: 1px solid var(--secondary-color);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .typewriter {
        overflow: hidden;
        border-right: .15em solid var(--accent-color);
        white-space: nowrap;
        letter-spacing: .15em;
        animation: typing 3.5s steps(40, end), blink-caret .75s step-end infinite;
    }
    
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    
    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: var(--accent-color); }
    }
</style>
""", unsafe_allow_html=True)

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load("tensflow.joblib")
    except:
        st.warning("Model file not found. Using a placeholder for demonstration.")
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor()

model = load_model()

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

# Car types data with placeholder images
car_types = [
    {
        "name": "Sedan",
        "image": "https://png.pngtree.com/png-clipart/20220921/ourmid/pngtree-mobil-sedan-merah-png-image_6207797.png",
        "description": "Sedans are four-door passenger cars with a separate trunk compartment. They typically offer comfortable seating for 4-5 passengers with good fuel efficiency and a smooth ride."
    },
    {
        "name": "SUV",
        "image": "https://png.pngtree.com/png-vector/20230212/ourmid/pngtree-red-suv-car-vector-png-image_6593674.png",
        "description": "Sport Utility Vehicles combine elements of passenger cars with features from off-road vehicles. They offer higher seating position, more cargo space, and often come with all-wheel drive capability."
    },
    {
        "name": "Hatchback",
        "image": "https://png.pngtree.com/png-vector/20230206/ourmid/pngtree-red-car-side-view-vector-illustration-in-trendy-flat-style-isolated-png-image_6585620.png",
        "description": "Hatchbacks are compact cars with a rear door that opens upward, offering versatile cargo space. They're fuel-efficient, easy to park, and popular in urban environments."
    },
    {
        "name": "Luxury",
        "image": "https://png.pngtree.com/png-vector/20230416/ourmid/pngtree-red-cabriolet-front-view-luxury-car-icon-vintage-flat-color-vector-png-image_6686793.png",
        "description": "Luxury cars focus on comfort, performance, and cutting-edge technology. They typically feature premium materials, advanced safety features, and powerful engines."
    },
    {
        "name": "Electric",
        "image": "https://cdn-icons-png.flaticon.com/512/6810/6810772.png",
        "description": "Electric vehicles run on electricity stored in rechargeable batteries. They produce zero emissions, have lower operating costs, and offer instant torque for responsive acceleration."
    }
]

# Sidebar content
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="https://thumbs.dreamstime.com/b/super-vector-car-logo-design-illustration-sleek-scalable-representing-automotive-speed-precision-modern-aesthetics-ideal-333985488.jpg" width="150" onerror="this.onerror=null; this.src='https://thumbs.dreamstime.com/b/super-vector-car-logo-design-illustration-sleek-scalable-representing-automotive-speed-precision-modern-aesthetics-ideal-333985488.jpg'">
        <h1 style="margin-top: 10px; font-size: 2rem;">AutoVault</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p>AutoVault is your advanced car valuation platform powered by machine learning. Our AI-driven system analyzes various parameters to predict the optimal selling price for your vehicle based on current market trends, vehicle specifications, and location factors.</p>
        <p>Use our intelligent prediction tool to get accurate estimates before making any buying or selling decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Quick Tips")
    st.info("✓ Fill all details for the most accurate prediction")
    st.info("✓ Check the Car Types tab to understand vehicle classifications")
    st.info("✓ Use the Expert Advice tab for personalized guidance")

# Main content
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🚗 AutoVault: Precision Car Valuation</h1>", unsafe_allow_html=True)

# Create tabs
tabs = st.tabs(["💰 Price Prediction", "🚙 Car Types", "💬 Expert Advice"])

# Tab 1: Price Prediction
with tabs[0]:
    st.markdown("<h2>Car Sale Price Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<p>Enter your car details below to get an estimated market value</p>", unsafe_allow_html=True)
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        #
        brand = st.selectbox('Brand', options=list(brand_mapping.keys()))
        year = st.number_input("Car Age (years)", min_value=1, max_value=50, value=5, step=1)
        km = st.number_input("Kilometers Driven", min_value=0, max_value=300000, value=50000, step=1000)
        fuel_type = st.selectbox('Fuel Type', options=list(fuel_type_mapping.keys()))
        transmission = st.selectbox('Transmission', options=list(transmission_mapping.keys()))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        
        owner_type = st.selectbox('Owner Type', options=list(owner_type_mapping.keys()))
        location = st.selectbox('Location', options=list(location_mapping.keys()))
        mileage = st.number_input("Mileage (km/l)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
        cc = st.number_input("Engine Capacity (CC)", min_value=500, max_value=8000, value=1500, step=50)
        power = st.number_input("Power (bhp)", min_value=0, max_value=1000, value=100, step=10)
        seat = st.number_input("Number of Seats", min_value=2, max_value=8, value=5, step=1)
        st.markdown("</div>", unsafe_allow_html=True)
    
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
    if st.button('Calculate Estimated Value', key='predict_button'):
        with st.spinner('Processing data...'):
            # Add a small delay for effect
            time.sleep(1)
            
            # Prepare input data
            input_data = prepare_input_data()

            try:
                # Make prediction using the model
                predicted_value = model.predict(input_data)
                
                # Ensure the predicted value is a numeric type and non-negative
                predicted_value = abs(float(predicted_value[0] if isinstance(predicted_value, np.ndarray) else predicted_value))
                
                # Show the predicted sale price with animation
                
                st.markdown("<h3>Estimated Market Value:</h3>", unsafe_allow_html=True)
                
                # Display with animation
                placeholder = st.empty()
                for i in range(5):
                    placeholder.markdown(f"<h2 style='color: #00FF41;'>₹{predicted_value*(0.85+random.random()*0.3):,.2f}</h2>", unsafe_allow_html=True)
                    time.sleep(0.2)
                
                placeholder.markdown(f"<h2 style='color: #00FF41;'>₹{predicted_value:,.2f}</h2>", unsafe_allow_html=True)
                
                st.markdown("<p>This prediction is based on current market analysis and the details you provided.</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Market insights based on the prediction
                st.markdown("<h3>Market Insights</h3>", unsafe_allow_html=True)
                
                # Generate some insights (these would be more data-driven in a real application)
                insights = [
                    f"{brand} vehicles in {location} have steady demand currently.",
                    f"{fuel_type} cars with {transmission} transmission are trending in the market.",
                    f"Cars with {power}bhp power and {mileage}km/l mileage are in the optimal range for resale value."
                ]
                
                for insight in insights:
                    st.info(insight)
                
            except Exception as e:
                st.error(f"Prediction error: {e}. Please check your inputs and try again.")

# Tab 2: Car Types
with tabs[1]:
    st.markdown("<h2>Car Types Guide</h2>", unsafe_allow_html=True)
    st.markdown("<p>Explore different types of cars and their characteristics</p>", unsafe_allow_html=True)
    
    # Display car types in a grid
    for i in range(0, len(car_types), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(car_types):
                
                st.image(car_types[i]["image"], caption=car_types[i]["name"])
                st.markdown(f"<h3>{car_types[i]['name']}</h3>", unsafe_allow_html=True)
                st.write(car_types[i]["description"])
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if i+1 < len(car_types):
                
                st.image(car_types[i+1]["image"], caption=car_types[i+1]["name"])
                st.markdown(f"<h3>{car_types[i+1]['name']}</h3>", unsafe_allow_html=True)
                st.write(car_types[i+1]["description"])
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional information about car classification
    
    st.markdown("<h3>Understanding Car Classifications</h3>", unsafe_allow_html=True)
    st.write("""
    Car classifications are based on various factors including body style, size, engine type, and intended use.
    When searching for a car, understanding these classifications can help you find a vehicle that best fits your needs
    in terms of space, fuel efficiency, performance, and budget.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 3: Expert Advice
with tabs[2]:
    st.markdown("<h2>AI Laptop Expert</h2>", unsafe_allow_html=True)
    st.markdown("<p>Get personalized advice for your laptop needs</p>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Set up Gemini API - You'll need to add your API key in the code or environment variables
    try:
        # Get API key from environment variable
        api_key = st.secrets["GEMINI_API_KEY"]
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            is_configured = True
        else:
            # Fallback message - this will only be visible to developers, not end users
            st.info("Note to developer: Set the GEMINI_API_KEY environment variable")
            is_configured = False
            model = None
    except Exception as e:
        st.error("Error initializing AI assistant")
        is_configured = False
        model = None
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about laptops (e.g., 'Which laptop is best for gaming?')"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        if is_configured:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Prepare prompt with context
                context = """
                You are a laptop expert advisor. Provide helpful, accurate advice about laptops.
                Focus on the user's specific needs and interests. Your advice should be technical
                but easy to understand. Recommend specific features based on use cases like gaming,
                productivity, design work, etc. Keep responses concise and helpful.
                """
                
                try:
                    response = model.generate_content(context + "\n\nUser question: " + prompt)
                    
                    # Simulate typewriter effect
                    if hasattr(response, 'text'):
                        response_text = response.text
                        for chunk in response_text.split():
                            full_response += chunk + " "
                            time.sleep(0.05)  # Adjust speed as needed
                            message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        message_placeholder.markdown("Sorry, I couldn't generate a response.")
                except Exception as e:
                    message_placeholder.markdown(f"Error generating response: {str(e)}")
        else:
            with st.chat_message("assistant"):
                st.markdown("AI assistant is currently unavailable. Please try again later.")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("<p style='text-align: left;'>Made with ❤️ by Mainak</p>", unsafe_allow_html=True)

with footer_col2:
    st.markdown("<p style='text-align: center;'>© 2025 AutoVault</p>", unsafe_allow_html=True)

with footer_col3:
    st.markdown("<p style='text-align: right;'>v1.0.0</p>", unsafe_allow_html=True)
