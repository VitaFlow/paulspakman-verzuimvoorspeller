import streamlit as st
import pandas as pd
import joblib

st.write("📋 Kolommen in de dataset:", df.columns.tolist())

# Inladen data en modellen
df = pd.read_excel("hr_verzuim_dataset_50.xlsx")  # Bestand moet in je projectmap staan
clf_model = joblib.load("model_classification_streamlit13.pkl")
reg_model = joblib.load("model_regression_streamlit13.pkl")
model_features = joblib.load("model_features_streamlit13.pkl")

# Functie om input voor het model klaar te maken
def prepare_input(data):
    df_input = pd.get_dummies(data, drop_first=True)
    for col in model_features:
        if col not in df_input.columns:
            df_input[col] = 0
    return df_input[model_features]

# AI-voorspellingen toepassen op gehele dataset
df_pred = df.copy()
X = prepare_input(df_pred)
df_pred["Verzuimkans"] = clf_model.predict_proba(X)[:, 1]
df_pred["VerwachteVerzuimdagen"] = reg_model.predict(X)
df_pred["risicoscore"] = df_pred["Verzuimkans"] * df_pred["VerwachteVerzuimdagen"]

# Sidebarfilters
st.sidebar.title("📊 Filters")
afdeling_filter = st.sidebar.selectbox("Afdeling", ["Alle"] + sorted(df_pred["Afdeling"].unique()))
functie_filter = st.sidebar.selectbox("Functie", ["Alle"] + sorted(df_pred["Functie"].unique()))
contract_filter = st.sidebar.selectbox("Contracttype", ["Alle"] + sorted(df_pred["Contracttype"].unique()))

# Filter toepassen op df_pred (en dus niet op df!)
df_filtered = df_pred.copy()
if afdeling_filter != "Alle":
    df_filtered = df_filtered[df_filtered["Afdeling"] == afdeling_filter]
if functie_filter != "Alle":
    df_filtered = df_filtered[df_filtered["Functie"] == functie_filter]
if contract_filter != "Alle":
    df_filtered = df_filtered[df_filtered["Contracttype"] == contract_filter]

# Kritieke medewerkers tonen
st.subheader("🚨 Kritieke medewerkers (hoogste risico)")
if "risicoscore" in df_filtered.columns:
    kritieke = df_filtered.sort_values("risicoscore", ascending=False)
    st.dataframe(kritieke[["Naam", "Afdeling", "Functie", "Verzuimkans", "VerwachteVerzuimdagen", "risicoscore"]])
else:
    st.warning("⚠️ 'risicoscore' kolom niet gevonden. Controleer of voorspelling is toegepast.")

# Individuele medewerker selecteren
st.subheader("👤 Detail per medewerker")
selected = st.selectbox("Selecteer medewerker:", df_filtered["Naam"])
record = df_filtered[df_filtered["Naam"] == selected]
st.write("📋 Medewerkergegevens:", record.T)

# Belangrijkste factoren
st.subheader("📈 Belangrijkste AI-factoren")
feature_importances = pd.Series(clf_model.feature_importances_, index=model_features)
st.bar_chart(feature_importances.sort_values(ascending=False).head(5))

