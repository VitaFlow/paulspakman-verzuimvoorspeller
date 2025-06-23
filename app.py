import streamlit as st
import pandas as pd

st.title("🔍 VerzuimVoorspeller Dashboard")

df = pd.read_excel("verzuim_voorspeller_prototype.xlsx", header=0)

df.columns = df.columns.str.strip().str.lower()

st.write("📋 Kolommen in de dataset:", df.columns.tolist())

if "functie" in df.columns:
    functie_filter = st.selectbox("Selecteer functie:", ["Alle"] + sorted(df["functie"].unique()))
else:
    st.error("❌ Kolom 'functie' niet gevonden in het Excelbestand.")
    functie_filter = "Alle"

if "leeftijd" in df.columns:
    leeftijd_filter = st.slider("Leeftijdsfilter", 25, 60, (25, 60))
else:
    st.error("❌ Kolom 'leeftijd' niet gevonden in het Excelbestand.")
    leeftijd_filter = (25, 60)

filtered_df = df.copy()
if functie_filter != "Alle" and "functie" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["functie"] == functie_filter]
if "leeftijd" in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df["leeftijd"] >= leeftijd_filter[0]) &
        (filtered_df["leeftijd"] <= leeftijd_filter[1])
    ]

st.subheader("📊 Overzicht met risicoscores")
if {"naam", "leeftijd", "functie", "totaalverzuim", "aantalverzuimmomenten", "laatsteverzuimdatum", "risicoscore", "risiconiveau"}.issubset(filtered_df.columns):
    st.dataframe(filtered_df[[
        "naam", "leeftijd", "functie", "totaalverzuim", "aantalverzuimmomenten",
        "laatsteverzuimdatum", "risicoscore", "risiconiveau"
    ]])
else:
    st.warning("⚠️ Niet alle verwachte kolommen aanwezig om de tabel te tonen.")

if "risiconiveau" in filtered_df.columns:
    st.subheader("📉 Risicoverdeling")
    st.bar_chart(filtered_df["risiconiveau"].value_counts())
else:
    st.warning("⚠️ Kolom 'risiconiveau' niet gevonden.")

