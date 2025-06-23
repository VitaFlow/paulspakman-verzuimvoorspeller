
import streamlit as st
import pandas as pd
import joblib

# Modellen en data inladen
df = pd.read_excel("verzuimdata_simulatie_10jaar.xlsx")
clf_model = joblib.load("model_classification.pkl")
reg_model = joblib.load("model_regression.pkl")
model_features = joblib.load("model_features.pkl")

# Voorbereiden features
def prepare_input(data):
    df_input = pd.get_dummies(data, drop_first=True)
    for col in model_features:
        if col not in df_input.columns:
            df_input[col] = 0
    return df_input[model_features]

# Streamlit dashboard
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

# Uitlegbaarheid (vereenvoudigd)
st.subheader("📊 Invloedrijke factoren (vereenvoudigd)")
feature_importances = pd.Series(clf_model.feature_importances_, index=model_features)
top_factors = feature_importances.sort_values(ascending=False).head(5)
st.bar_chart(top_factors)

