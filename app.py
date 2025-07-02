import streamlit as st
import pandas as pd
import joblib

# Data en modellen inladen
df = pd.read_excel("verzuimdata_hr_50.xlsx")
clf_model = joblib.load("model_classification.pkl")
reg_model = joblib.load("model_regression.pkl")
model_features = joblib.load("model_features.pkl")

# Input voorbereiden
def prepare_input(data):
    df_input = pd.get_dummies(data, drop_first=True)
    for col in model_features:
        if col not in df_input.columns:
            df_input[col] = 0
    return df_input[model_features]

st.title("🧠 AI Verzuimvoorspeller")

# 🔽 Medewerkers sorteren op risicoscore (hoog naar laag)
if "risicoscore" in df.columns:
    df_sorted = df.sort_values(by="risicoscore", ascending=False)
else:
    st.error("❌ Kolom 'risicoscore' niet gevonden.")
    st.stop()

# 🔔 Kritieke melding tonen
kritiek = df_sorted[df_sorted["risiconiveau"] == "hoog"]
if not kritiek.empty:
    st.warning(f"⚠️ {len(kritiek)} medewerkers met hoog verzuimrisico!")
    st.dataframe(kritiek[["Naam", "functie", "risicoscore", "risiconiveau"]])

# 🔎 Selectie via dropdown gesorteerd op risico
selected = st.selectbox("Selecteer medewerker (gesorteerd op risico):", df_sorted["Naam"])
record = df_sorted[df_sorted["Naam"] == selected]
st.write("📋 Medewerkergegevens:", record.T)

# Voorspellingen uitvoeren
X_input = prepare_input(record)
verzuimkans = clf_model.predict_proba(X_input)[0][1]
verwachte_dagen = reg_model.predict(X_input)[0]

# Resultaten tonen
st.subheader("🔮 Voorspellingen")
st.metric("Kans op verzuim", f"{verzuimkans:.0%}")
st.metric("Verwachte verzuimdagen", f"{verwachte_dagen:.1f} dagen")
