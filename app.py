import streamlit as st
import pandas as pd
import joblib

# --- Bestanden inladen ---
df = pd.read_excel("hr_verzuim_dataset_50.xlsx")  # Zorg dat dit bestand in je map staat
clf_model = joblib.load("model_classification.pkl")
reg_model = joblib.load("model_regression.pkl")
model_features = joblib.load("model_features.pkl")

# --- Input voorbereiden ---
def prepare_input(data):
    df_input = pd.get_dummies(data, drop_first=True)
    for col in model_features:
        if col not in df_input.columns:
            df_input[col] = 0
    return df_input[model_features]

# --- Streamlit UI ---
st.set_page_config("AI Verzuimvoorspeller", layout="wide")
st.title("🧠 AI Verzuimvoorspeller voor HR")
st.caption("Bekijk risico's, voorspellingen en advies per medewerker")

# --- Filter: Afdeling (of rol/functie als je dat wilt aanpassen) ---
afdelingen = df["Afdeling"].dropna().unique().tolist()
afdeling_keuze = st.selectbox("📂 Filter op afdeling", ["Alle"] + afdelingen)
if afdeling_keuze != "Alle":
    df_filtered = df[df["Afdeling"] == afdeling_keuze]
else:
    df_filtered = df.copy()

# --- Sorteren op hoogste risicoscore ---
kritieke = df_filtered.sort_values("risicoscore", ascending=False)

# --- Kritieke medewerkers tabel bovenaan ---
st.subheader("🚨 Kritieke medewerkers (hoogste risico eerst)")
st.dataframe(kritieke[["Naam", "Afdeling", "risicoscore", "risiconiveau"]].head(5), use_container_width=True)

# --- Medewerkerselectie ---
st.subheader("🔍 Bekijk details per medewerker")
medewerker = st.selectbox("Selecteer een medewerker", kritieke["Naam"])
record = df[df["Naam"] == medewerker]

st.write("📋 Gegevens", record.T)

# --- Voorspellingen ---
X_input = prepare_input(record)
verzuimkans = clf_model.predict_proba(X_input)[0][1]
verwachte_dagen = reg_model.predict(X_input)[0]

st.subheader("📈 Voorspellingen")
st.metric("Kans op verzuim", f"{verzuimkans:.0%}")
st.metric("Verwachte verzuimdagen", f"{verwachte_dagen:.1f} dagen")

# --- Uitlegbaarheid ---
st.subheader("📊 Belangrijkste voorspelfactoren")
importances = pd.Series(clf_model.feature_importances_, index=model_features)
top_5 = importances.sort_values(ascending=False).head(5)
st.bar_chart(top_5)

