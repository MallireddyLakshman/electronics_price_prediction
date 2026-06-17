import streamlit as st
import pickle
import pandas as pd
st.set_page_config(page_title="Electronics Price Prediction")

st.title("📱📺❄️ Electronics Price Prediction")

category = st.selectbox(
    "Select Product Category",
    ["Mobile", "TV", "Air Conditioner"]
)

if category == "Mobile":
    st.header("📱 Mobile Price Prediction")

    brand = st.selectbox(
        "Brand",
        ["Samsung", "Apple", "Xiaomi", "OnePlus", "Realme", "Vivo", "Oppo", "Motorola"]
    )

    ram = st.number_input("RAM (GB)", min_value=2, max_value=16, value=8)
    rom = st.number_input("Storage (GB)", min_value=32, max_value=512, value=128)
    battery = st.number_input("Battery (mAh)", min_value=2000, max_value=7000, value=5000)
    camera = st.number_input("Camera (MP)", min_value=8, max_value=200, value=50)

if st.button("Predict Mobile Price"):

    model = pickle.load(open("models/mobile_model.pkl", "rb"))
    encoder = pickle.load(open("models/mobile_encoder.pkl", "rb"))

    brand_encoded = encoder.transform([brand])[0]

    input_data = pd.DataFrame([{
        "Brand": brand_encoded,
        "RAM": ram,
        "ROM": rom,
        "Battery": battery,
        "Camera": camera
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Mobile Price: ₹{prediction:,.0f}")

elif category == "TV":
    st.header("📺 TV Price Prediction")

    brand = st.selectbox(
        "Brand",
        ["Samsung", "LG", "Sony", "Mi", "OnePlus", "TCL", "Panasonic"]
    )

    screen = st.number_input("Screen Size (inches)", min_value=24, max_value=100, value=43)
    resolution = st.selectbox("Resolution", ["Full HD", "4K"])
    smart_tv = st.selectbox("Smart TV", ["Yes", "No"])

if st.button("Predict TV Price"):

    model = pickle.load(open("models/tv_model.pkl", "rb"))
    encoder = pickle.load(open("models/tv_encoder.pkl", "rb"))

    brand_encoded = encoder.transform([brand])[0]

    resolution_value = 4 if resolution == "4K" else 2
    smart_tv_value = 1 if smart_tv == "Yes" else 0

    input_data = pd.DataFrame([{
        "Brand": brand_encoded,
        "Screen_Size": screen,
        "Resolution": resolution_value,
        "Smart_TV": smart_tv_value
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated TV Price: ₹{prediction:,.0f}")

else:
    st.header("❄️ Air Conditioner Price Prediction")

    brand = st.selectbox(
        "Brand",
        ["LG", "Samsung", "Daikin", "Voltas", "BlueStar", "Hitachi", "Whirlpool"]
    )

    tonnage = st.selectbox("Tonnage", [1.0, 1.5, 2.0])
    star = st.selectbox("Star Rating", [3, 4, 5])
    inverter = st.selectbox("Inverter", ["Yes", "No"])

if st.button("Predict AC Price"):

    model = pickle.load(open("models/ac_model.pkl", "rb"))
    encoder = pickle.load(open("models/ac_encoder.pkl", "rb"))

    brand_encoded = encoder.transform([brand])[0]

    inverter_value = 1 if inverter == "Yes" else 0

    input_data = pd.DataFrame([{
        "Brand": brand_encoded,
        "Tonnage": tonnage,
        "Star_Rating": star,
        "Inverter": inverter_value
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated AC Price: ₹{prediction:,.0f}")