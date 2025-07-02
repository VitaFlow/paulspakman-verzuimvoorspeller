import streamlit as st
import pandas as pd
import joblib

# Data inladen
bestand = "verzuimdata_met_extra_features.xlsx"
df = pd.read_excel(bestand)

# Nieuwe kolom toevoegen: dagen sinds laatste verzuim
if 'laatsteverzuimdatum' in df.columns:
    df['dagen_sinds_verzuim'] = (pd.Timestamp.today() - pd.to_datetime(df['laatsteverzuimdatum'])).dt.days

# Features instellen
features = [
    "leeftijd", "geslacht", "functie", "contracttype", "uren_per_week",
    "thuiswerken", "mentale_belasting", "fysieke_belasting", "werkstress", "tevredenheid",
    "aantal_verzuimmomenten", "duur_in_dienst", "leidinggevende", "sector",
    "dagen_sinds_verzuim"
]

# Modellen inladen
clf_model = joblib.load("model_classification_streamlit13.pkl")
reg_model = joblib.load("model_regression_streamlit13.pkl")

# Functie om input klaar te maken voor predictie
def prepare_input(data):
    df_input = pd.get_dummies(data, drop_first=True)
    for col in features:
        if col not in df_input.columns:
            df_input[col] = 0
    return df_input[features]

# Streamlit UI
st.title("🧠 AI Verzuimvoorspeller")

selected = st.selectbox("Selecteer medewerker:", df["Naam"])
record = df[df["Naam"] == selected]
st.write("📋 Medewerkergegevens:", record.T)

X_input = prepare_input(record)

# Voorspellingen
verzuimkans = clf_model.predict_proba(X_input)[0][1]
verwachte_dagen = reg_model.predict(X_input)[0]

st.subheader("🔮 Voorspellingen")
st.metric("Kans op verzuim (komend jaar)", f"{verzuimkans:.0%}")
st.metric("Verwachte verzuimdagen", f"{verwachte_dagen:.1f} dagen")

# Belangrijkste factoren
st.subheader("📊 Invloedrijke factoren (vereenvoudigd)")
feature_importances = pd.Series(clf_model.feature_importances_, index=features)
top_factors = feature_importances.sort_values(ascending=False).head(5)
st.bar_chart(top_factors)
