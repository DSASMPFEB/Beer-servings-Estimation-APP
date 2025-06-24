import pickle
import streamlit as st
import numpy as np
from os import path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Beer Servings Estimation App")

df = pd.read_csv("data/beer-servings.csv") 

st.subheader("Alcohol Consumption Overview")

avg_servings = df.groupby("continent")[
    ["beer_servings", "wine_servings", "spirit_servings"]].mean()

st.markdown("**Average Alcohol Servings by Continent**")
st.bar_chart(avg_servings)


fig, ax = plt.subplots()
sns.scatterplot(
    data=df, x="beer_servings", y="total_litres_of_pure_alcohol", hue="continent", ax=ax)
st.markdown("**BEER SERVINGS vs TOTAL ALCOHOL CONSUMPTION**")
st.pyplot(fig)

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
