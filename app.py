import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "gender_classification_thinking.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)


st.title("Gender Detection Using Genral Questions")

# Favorite Color,Favorite Music Genre,Favorite Beverage,Favorite Soft Drink

with st.form("gender_form"):
    
    fav_col = st.selectbox(
        "Select your Favourite Color : ",
        options = ['Cool', 'Neutral', 'Warm']
    )
    
    fav_music = st.selectbox(
        "Select your Favourite Music Genere : ",
        options=['Rock', 'Hip hop', 'Folk/Traditional', 'Jazz/Blues', 'Pop', 'Electronic', 'R&B and soul']
    )
    
    fav_beverage = st.selectbox(
        "Select your Favourite Beverages : ",
        options=['Vodka', 'Wine', 'Whiskey', "Doesn't drink", 'Beer', 'Other']
    )
    
    fav_soft_drink = st.selectbox(
        "Select your Favourite Soft Drink : ",
        options=['7UP/Sprite', 'Coca Cola/Pepsi', 'Fanta', 'Other']
    )
    
    submit = st.form_submit_button(label="Predict (M/F)")
    

if submit:
    data = [fav_col, fav_music, fav_beverage, fav_soft_drink]
    
    prediction = model.predict([data])
    
    print(prediction)
    # st.write(prediction)
    gender = ""
    if prediction[0] : 
        gender = "Male"
    else:
        gender = "Female"
        # st.write("Female")
    st.markdown(
        f"""
        <div style="
            background-color: #0E1117; 
            padding: 15px; 
            border-radius: 10px; 
            # border: 2px solid #ddd; 
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
            ">
            <h1 style="color: #333; text-align: center;">I Guss Your Gender Is </h1>
            <h2 style="color: #555; text-align: center;">{gender}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
