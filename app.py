import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 Real Estate Investment Advisor")
st.write("Predict property investment potential using Machine Learning.")

import os
import joblib
import streamlit as st

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "investment_classifier.joblib"
)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found in the repository.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    state = st.text_input("State", "Maharashtra")
    city = st.text_input("City", "Nashik")
    locality = st.text_input("Locality", "City Center")
    property_type = st.selectbox(
        "Property Type", ["Apartment", "House", "Villa", "Other"]
    )
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
    size = st.number_input("Size (sq. ft.)", min_value=100, value=1000)
    price = st.number_input("Price (lakhs)", min_value=1.0, value=50.0)

with col2:
    year_built = st.number_input(
        "Year Built", min_value=1950, max_value=2026, value=2020
    )
    furnished = st.selectbox(
        "Furnished Status", ["Furnished", "Semi-Furnished", "Unfurnished"]
    )
    floor_no = st.number_input("Floor Number", min_value=0, value=1)
    total_floors = st.number_input("Total Floors", min_value=1, value=5)
    schools = st.number_input("Nearby Schools", min_value=0, value=3)
    hospitals = st.number_input("Nearby Hospitals", min_value=0, value=2)
    transport = st.selectbox(
        "Public Transport Accessibility", ["High", "Medium", "Low"]
    )

parking = st.selectbox("Parking Space", ["Yes", "No"])
security = st.selectbox("Security", ["Yes", "No"])
amenities = st.text_input("Amenities", "Gym, Lift")
facing = st.selectbox(
    "Facing", ["North", "South", "East", "West", "Other"]
)
owner_type = st.selectbox(
    "Owner Type", ["Owner", "Builder", "Agent", "Other"]
)
availability = st.selectbox(
    "Availability Status", ["Ready to Move", "Under Construction", "Other"]
)

if st.button("Predict Investment Potential"):
    current_year = 2026

    input_df = pd.DataFrame([{
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Price_in_Lakhs": price,
        "Price_per_SqFt": price * 100000 / size,
        "Year_Built": year_built,
        "Furnished_Status": furnished,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Age_of_Property": current_year - year_built,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Amenities": amenities,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability
    }])

    try:
        prediction = model.predict(input_df)[0]
        st.subheader("Investment Prediction")
        st.success(f"Predicted Investment Label: {prediction}")
    except Exception as e:
        st.error("The input columns or categories do not match the trained model.")
        st.code(str(e))
