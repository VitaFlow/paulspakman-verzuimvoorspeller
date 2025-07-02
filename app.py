import streamlit as st
import pandas as pd
import joblib

# --- DATA & MODELLEN INLADEN ---
df = pd.read_excel("hr_verzuim_dataset_50.xlsx")  # dataset met 50 medewerkers en HR-features
clf_model = joblib.load("model_classification.pkl")
reg_model = joblib.load("model_regression.pkl")
model_features = joblib.load("model_features.pkl")

# --- FEATURE PREPARATIE ---
def prepare_input(data):
    df_input = pd.get_dummies(data, drop_first=True)
    for col in model_features:
        if col not in df_input.columns:
            df_input[col] = 0
    return df_input[model_features]

# --- TITEL EN UITLEG ---
st.title("🧠 AI Verzuimvoorspeller voor HR")
st.markdown("Krijg direct inzicht in medewerkers met verhoogd verzuimrisico.")

# --- FILTERS ---
afdelingen = df["Afdeling"].unique()
afdeling_filter = st.selectbox("📂 Filter op afdeling:", ["Alle"] + list(afdelingen))
if afdeling_filter != "Alle":
    df = df[df["Afdeling"] == afdeling_filter]

# --- VERZUIM SCORE BEREKENEN ---
input_data = prepare_input(df)
df["Verzuimkans"] = clf_model.predict_proba(input_data)[:, 1]
df["Verwachte verzuimdagen"] = reg_model.predict(input_data)

# --- KRITIEKE WERKNEMERS BOVENAAN ---
df_sorted = df.sort_values(by="Verzuimkans", ascending=False).reset_index(drop=True)
kritiek = df_sorted[df_sorted["Verzuimkans"] > 0.6]

if not kritiek.empty:
    st.subheader("🚨 Kritieke medewerkers met hoog risico")
    st.dataframe(kritiek[["Naam", "Afdeling", "Verzuimkans", "Verwachte verzuimdagen"]])
else:
    st.info("Geen medewerkers met verhoogd risico gevonden.")

# --- SELECTEER INDIVIDU ---
st.subheader("👤 Individuele analyse")
medewerker = st.selectbox("Selecteer een medewerker:", df_sorted["Naam"])
record = df_sorted[df_sorted["Naam"] == medewerker]
X_input = prepare_input(record)

verzuimkans = clf_model.predict_proba(X_input)[0][1]
verwachte_dagen = reg_model.predict(X_input)[0]

st.metric("📊 Kans op verzuim (komend jaar)", f"{verzuimkans:.0%}")
st.metric("📅 Verwachte verzuimdagen", f"{verwachte_dagen:.1f} dagen")

# --- TOON PROFIEL ---
with st.expander("📋 Bekijk medewerkergegevens"):
    st.write(record.T)

# --- UITLEGBAARHEID ---
st.subheader("📈 Belangrijkste risicofactoren")
importances = pd.Series(clf_model.feature_importances_, index=model_features)
top_factors = importances.sort_values(ascending=False).head(5)
st.bar_chart(top_factors)

