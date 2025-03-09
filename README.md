# 🚗 AutoMart - Predict Your Car Worth

**AutoMart** is a powerful machine learning web app that predicts the sale value of used cars! By leveraging a deep learning model built with **TensorFlow**, this app estimates car prices based on various input features like brand, location, mileage, engine specifications, and more. With **Streamlit** as the frontend, the app provides a seamless and interactive user experience to estimate the value of any used car in real-time.

---

## 📦 **Technologies Used**

- **Python**: The main programming language used to build the app.
- **TensorFlow**: Deep learning framework for training the car price prediction model.
- **Streamlit**: Interactive web framework for building and deploying the user interface.
- **Scikit-learn**: Used for preprocessing the data and splitting it into training and testing sets.
- **Pandas & NumPy**: For efficient data manipulation and numerical computations.
- **Plotly**: For visualizing model training progress and evaluation metrics.

---


## 🚀 **Usage**

1. **Run the Streamlit app**:

   Launch the app by clicking:

   
2. **Enter Car Details**:

   Use the intuitive UI to input details about the car:
   - **Location**: Choose the car’s location from a dropdown.
   - **Fuel Type**: Select the car's fuel type (e.g., Petrol, Diesel, Electric).
   - **Transmission**: Choose between Automatic or Manual transmission.
   - **Owner Type**: Select whether the car is being sold by the first, second, or third owner.
   - **Car Features**: Enter the car's year of manufacture, mileage, engine capacity (CC), power (bhp), and number of seats.

3. **Get Predicted Price**:

   After entering the details, click **"Predict Car Sale Value"** and watch the app instantly generate a predicted sale price for the car!

---


## 🔧 **Model Overview**

The car sale price prediction model is built using a **Multi-Layer Perceptron (MLP)** regressor neural network. Here’s a breakdown of the model:

- **Model Type**: MLP Regressor (Deep Neural Network).
- **Loss Function**: Mean Squared Error (MSE) for regression tasks.
- **Metrics**: Model evaluation is done using MAE (Mean Absolute Error), MSE, RMSE, and R² score.

---

## 🗂️ **Project Structure**

Here’s how the project is organized:

```
├── app.py                  # Streamlit app (main entry point)
├── model                   # Folder containing saved model files (e.g., tensflow.joblib)
├── requirements.txt        # List of required Python libraries
├── README.md               # This file (project description and setup guide)
├── train-data.csv          # Dataset used for training the model
├── test-data.csv           # Dataset used for testing the model
└── utils                   # Utility functions for data preprocessing
```

---

## 📊 **Model Evaluation**

After training the model, we evaluate its performance using the following metrics:

- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in the predictions.
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: Square root of MSE, a more interpretable version.
- **R² Score**: Indicates how well the model explains the variance in the target data (Car Sale Price).

---

## 🧑‍💻 **Contributing**

We welcome contributions! Feel free to:
- **Fork** the repository.
- **Make changes** or improvements to the code.
- **Submit a pull request** with your changes.
