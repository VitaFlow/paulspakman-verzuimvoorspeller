import streamlit as st
import pandas as pd
import joblib

# Data en modellen inladen
df = pd.read_excel("hr_verzuim_dataset_50.xlsx")
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

# Titel
st.title("🧠 AI Verzuimvoorspeller")

# Filters voor HR
st.sidebar.header("🔍 Filter medewerkers")
afdelingen = st.sidebar.multiselect("Afdeling", options=df["Afdeling"].unique(), default=list(df["Afdeling"].unique()))
functies = st.sidebar.multiselect("Functie", options=df["Functie"].unique(), default=list(df["Functie"].unique()))
contracttypes = st.sidebar.multiselect("Contracttype", options=df["Contracttype"].unique(), default=list(df["Contracttype"].unique()))

df_filtered = df[
    df["Afdeling"].isin(afdelingen) &
    df["Functie"].isin(functies) &
    df["Contracttype"].isin(contracttypes)
].copy()

# Voorspellingen toevoegen aan DataFrame
verzuimkansen = []
verwachte_dagen = []

for i, row in df_filtered.iterrows():
    input_row = prepare_input(pd.DataFrame([row]))
    kans = clf_model.predict_proba(input_row)[0][1]
    dagen = reg_model.predict(input_row)[0]
    verzuimkansen.append(kans)
    verwachte_dagen.append(dagen)

df_filtered["Verzuimkans"] = verzuimkansen
df_filtered["Verwachte dagen"] = verwachte_dagen
df_filtered["risicoscore"] = df_filtered["Verzuimkans"] * df_filtered["Verwachte dagen"]

# 🚨 Kritieke medewerkers tonen
st.subheader("🚨 Kritieke medewerkers (hoogste risico eerst)")
kritieke = df_filtered.sort_values("risicoscore", ascending=False).head(10)
st.dataframe(kritieke[["Naam", "Afdeling", "Functie", "Verzuimkans", "Verwachte dagen", "risicoscore"]])

# 📋 Individuele voorspelling
st.subheader("📋 Bekijk individuele medewerker")
selected = st.selectbox("Selecteer medewerker:", df_filtered["Naam"])
record = df_filtered[df_filtered["Naam"] == selected]
st.write("🧾 Medewerkergegevens:", record.T)

# 📊 Simpele uitleg topfactoren (optioneel, indien gewenst)
st.subheader("📊 Belangrijkste voorspelfactoren (algemeen)")
feature_importances = pd.Series(clf_model.feature_importances_, index=model_features)
top_factors = feature_importances.sort_values(ascending=False).head(5)
st.bar_chart(top_factors)

