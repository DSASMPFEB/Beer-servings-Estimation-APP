import pickle
import streamlit as st
import numpy as np
from os import path

st.title("Beer Servings Estimation App")


file_name = "lr_reg.pkl"
with open(path.join("model", file_name), "rb") as f:
    lr_model = pickle.load(f)


country = st.selectbox(
    "Select a country",
    ["Germany", "USA", "India", "Brazil", "Czech Republic", "Ireland", "Japan"],
)

continent = st.selectbox(
    "Select a continent",
    ["Europe", "North America", "Asia", "South America", "Africa", "Oceania"],
)

beer = st.number_input("beer servings", min_value=0)
spirit = st.number_input("spirit servings", min_value=0)
wine = st.number_input("wine servings", min_value=0)


if st.button("Predict"):
    features = np.array([[beer, spirit, wine]])
    pred = lr_model.predict(features)
    st.markdown(f"**Estimated total liters of pure alcohol: {round(pred[0], 2)}**")
