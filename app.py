import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np


model=pickle.load(open("LassoModel.pkl",'rb'))

df = pd.read_csv("Cleaned_data.csv")

st.title("Bengaluru House Price Prediction")
st.markdown("Want to predict the price of new House in Bengaluru ? Try filling the details below")
st.subheader("select the features that you want")
location=st.selectbox("select the location :",sorted(df['location'].unique()))
bhk=st.selectbox("Select the BHK :",sorted(df['bhk'].unique()))
bath=st.text_input("Enter the No of Bathrooms you want :")
balcony=st.text_input("Enter the No of balcony you want :")
total_sqft_int=st.text_input("Enter the total_sqft you want :")

if st.button("Predict Price"):
    # Preprocess the input features as needed
    features = pd.DataFrame({
        'location': [location],
        'bhk': [bhk],
        'bath': [bath],
        'balcony': [balcony],
        'total_sqft_int': [total_sqft_int],
    })

    # Make the prediction using the pre-trained model
    prediction = model.predict(features)

    # Display the predicted price
    st.success(f"Predicted House Price: lakhs {prediction[0]:,.2f}")