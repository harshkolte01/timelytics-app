# Import required libraries
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import requests
import os

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Timelytics", page_icon=":pencil:", layout="wide")

# Display title and captions
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - XGBoost, Random Forests, and Support Vector Machines (SVM) - to accurately forecast Order to Delivery (OTD) times. By combining the strengths of these three algorithms, Timelytics provides a robust and reliable prediction of OTD times, helping businesses to optimize their supply chain operations."
)

st.caption(
    "With Timelytics, businesses can identify potential bottlenecks and delays in their supply chain and take proactive measures to address them, reducing lead times and improving delivery times. The model utilizes historical data on order processing times, production lead times, shipping times, and other relevant variables to generate accurate forecasts of OTD times. These forecasts can be used to optimize inventory management, improve customer service, and increase overall efficiency in the supply chain."
)

# Download and load the trained ensemble model
modelfile = "voting_model.pkl"
model_url = "https://drive.google.com/uc?export=download&id=1ggFToZ1rv-O5xwMqskV3orm26i72Lmd0"

if not os.path.exists(modelfile):
    try:
        st.write("Downloading model file...")
        response = requests.get(model_url)
        response.raise_for_status()  # Raise an error for bad responses
        with open(modelfile, "wb") as f:
            f.write(response.content)
        st.write("Model downloaded successfully.")
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        st.stop()

try:
    voting_model = pickle.load(open(modelfile, "rb"))
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Cache the model for faster loading
@st.cache_resource
def load_model():
    return voting_model

# Define the wait time predictor function
def waitime_predictor(
    purchase_dow,
    purchase_month,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer,
    geolocation_state_seller,
    distance,
):
    try:
        prediction = voting_model.predict(
            np.array(
                [
                    [
                        purchase_dow,
                        purchase_month,
                        year,
                        product_size_cm3,
                        product_weight_g,
                        geolocation_state_customer,
                        geolocation_state_seller,
                        distance,
                    ]
                ]
            )
        )
        return round(prediction[0])
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Define input parameters in sidebar
with st.sidebar:
    # Uncomment the following lines if the image file exists
    # try:
    #     img = Image.open("./assets/supply_chain_optimisation.jpg")
    #     st.image(img)
    # except FileNotFoundError:
    #     st.warning("Image not found. Continuing without image.")
    
    st.header("Input Parameters")
    purchase_dow = st.number_input(
        "Purchased Day of the Week (0=Monday, 6=Sunday)",
        min_value=0, max_value=6, step=1, value=3,
        help="Enter 0 for Monday, 1 for Tuesday, ..., 6 for Sunday"
    )
    purchase_month = st.number_input(
        "Purchased Month", min_value=1, max_value=12, step=1, value=1,
        help="Enter 1 for January, 2 for February, ..., 12 for December"
    )
    year = st.number_input("Purchased Year", value=2018, help="Enter the year of purchase")
    product_size_cm3 = st.number_input(
        "Product Size in cm^3", value=9328, help="Enter product volume in cubic centimeters"
    )
    product_weight_g = st.number_input(
        "Product Weight in grams", value=1800, help="Enter product weight in grams"
    )
    geolocation_state_customer = st.number_input(
        "Geolocation State of Customer (e.g., state code)", value=10,
        help="Enter the numerical state code for the customer"
    )
    geolocation_state_seller = st.number_input(
        "Geolocation State of Seller (e.g., state code)", value=20,
        help="Enter the numerical state code for the seller"
    )
    distance = st.number_input(
        "Distance (km)", value=475.35, help="Enter distance between customer and seller in kilometers"
    )
    submit = st.button(label="Predict Wait Time!")

# Define output container
with st.container():
    st.header("Output: Wait Time in Days")
    if submit:
        prediction = waitime_predictor(
            purchase_dow,
            purchase_month,
            year,
            product_size_cm3,
            product_weight_g,
            geolocation_state_customer,
            geolocation_state_seller,
            distance,
        )
        if prediction is not None:
            with st.spinner(text="This may take a moment..."):
                st.success(f"Predicted Delivery Time: {prediction} days")

    # Sample dataset
    data = {
        "Purchased Day of the Week": ["0", "3", "1"],
        "Purchased Month": ["6", "3", "1"],
        "Purchased Year": ["2018", "2017", "2018"],
        "Product Size in cm^3": ["37206.0", "63714", "54816"],
        "Product Weight in grams": ["16250.0", "7249", "9600"],
        "Geolocation State Customer": ["25", "25", "25"],
        "Geolocation State Seller": ["20", "7", "20"],
        "Distance": ["247.94", "250.35", "4.915"],
    }
    df = pd.DataFrame(data)
    st.header("Sample Dataset")
    st.write(df)