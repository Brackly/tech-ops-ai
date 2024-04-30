import streamlit as st
import joblib
import numpy as np
import pandas as pd
from model import distances

model = joblib.load('model.joblib')

def predict_price( weight, parcel_type, delivery_service, is_holiday_or_weekend,previous_orders,distance):
                
    input_data = {
                  'parcel_weight': [weight],
                  'Parcel_type': [parcel_type],
                  'Delivery_Service': [delivery_service],
                  'is_holiday_or_weekend': [is_holiday_or_weekend],
                  'previous_orders':[previous_orders],
                  'Distance': [distance]}
    
    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df.values)
    return prediction[0]


def main():
    st.title("Parcel Delivery Prediction")

    origin = st.selectbox("Origin", ["Nairobi", "Mombasa", "Mandera"])
    destination = st.selectbox("Destination", ["Nairobi", "Mombasa", "Mandera"])
    parcel_weight  = st.number_input("Parcel Weight (kg)", min_value=0.5, step=0.5)
    Parcel_type = st.selectbox("Parcel Type", ["Delicate", "Normal"])
    Delivery_Service = st.selectbox("Delivery Service", ["Express", "Standard"])

    if st.button("Predict Delivery Time"):
        origin_dest=f"{origin}_{destination}"
        Distance=distances[origin_dest]
        Parcel_type =1 if Parcel_type=='Delicate' else 0
        Delivery_Service =1 if Delivery_Service=='Express' else 0
        is_holiday_or_weekend  = np.random.randint(0,1)
        previous_orders= np.random.randint(0,500)
        prediction = predict_price( parcel_weight , Parcel_type , Delivery_Service , is_holiday_or_weekend ,previous_orders,Distance)

        st.success(f"Estimated Price is: Ksh.{prediction}")

if __name__ == "__main__":
    main()
