import streamlit as st
import pandas as pd

# Dataset inladen
df = pd.read_excel("verzuim_voorspeller_prototype.xlsx")

st.title("🔍 VerzuimVoorspeller Dashboard")

# Filters
functie_filter = st.selectbox("Selecteer functie:", ["Alle"] + sorted(df["Functie"].unique()))
leeftijd_filter = st.slider("Leeftijdsfilter", 25, 60, (25, 60))

# Filter logica
filtered_df = df.copy()
if functie_filter != "Alle":
    filtered_df = filtered_df[filtered_df["Functie"] == functie_filter]
filtered_df = filtered_df[(filtered_df["Leeftijd"] >= leeftijd_filter[0]) & (filtered_df["Leeftijd"] <= leeftijd_filter[1])]

st.subheader("📋 Medewerkersoverzicht met Risicoscore")
st.dataframe(filtered_df[[
    "Naam", "Leeftijd", "Functie", "TotaalVerzuim", "AantalVerzuimMomenten", 
    "LaatsteVerzuimDatum", "RisicoScore", "RisicoNiveau"
]])

st.subheader("📊 Verzuimverdeling per Risiconiveau")
st.bar_chart(filtered_df["RisicoNiveau"].value_counts())
