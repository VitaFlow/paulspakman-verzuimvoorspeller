import streamlit as st
import pandas as pd
import joblib

# Laad modellen en features
df = pd.read_excel("hr_verzuim_dataset_50.xlsx")
clf_model = joblib.load("model_classification_streamlit13.pkl")
reg_model = joblib.load("model_regression_streamlit13.pkl")
model_features = joblib.load("model_features_streamlit13.pkl")

# Eén-hot encoding + missende features aanvullen
def prepare_input(data):
    df_input = pd.get_dummies(data, drop_first=True)
    for col in model_features:
        if col not in df_input.columns:
            df_input[col] = 0
    return df_input[model_features]

# Verwerk voorspellingen voor alle medewerkers
df_predictions = df.copy()
X_all = prepare_input(df_predictions)
df_predictions["Verzuimkans"] = clf_model.predict_proba(X_all)[:, 1]
df_predictions["Verwachte dagen"] = reg_model.predict(X_all)
df_predictions["risicoscore"] = df_predictions["Verzuimkans"] * df_predictions["Verwachte dagen"]

# Sidebar filters
st.sidebar.title("🔎 Filters")
afdelingen = ["Alle"] + sorted(df_predictions["Afdeling"].unique())
functies = ["Alle"] + sorted(df_predictions["Functie"].unique())
contracttypes = ["Alle"] + sorted(df_predictions["Contracttype"].unique())

afdeling_selectie = st.sidebar.selectbox("Afdeling", afdelingen)
functie_selectie = st.sidebar.selectbox("Functie", functies)
contract_selectie = st.sidebar.selectbox("Contracttype", contracttypes)

# Filter toepassen
df_filtered = df_predictions.copy()
if afdeling_selectie != "Alle":
    df_filtered = df_filtered[df_filtered["Afdeling"] == afdeling_selectie]
if functie_selectie != "Alle":
    df_filtered = df_filtered[df_filtered["Functie"] == functie_selectie]
if contract_selectie != "Alle":
    df_filtered = df_filtered[df_filtered["Contracttype"] == contract_selectie]

# Toon kritieke medewerkers
st.subheader("🚨 Kritieke medewerkers (hoogste risico eerst)")
kritieke = df_filtered.sort_values("risicoscore", ascending=False).head(10)
st.dataframe(kritieke[["Naam", "Afdeling", "Functie", "Verzuimkans", "Verwachte dagen", "risicoscore"]])

# Medewerkerselectie
st.subheader("👤 Details per medewerker")
geselecteerde = st.selectbox("Selecteer medewerker:", df_filtered["Naam"])
record = df_filtered[df_filtered["Naam"] == geselecteerde]

st.write("📋 Gegevens:", record.T)

# Uitlegbaarheid
st.subheader("📊 Invloedrijke factoren")
feature_importances = pd.Series(clf_model.feature_importances_, index=model_features)
top_factors = feature_importances.sort_values(ascending=False).head(5)
st.bar_chart(top_factors)

