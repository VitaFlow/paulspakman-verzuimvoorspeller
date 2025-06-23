
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Data inladen
df = pd.read_excel("verzuimdata_simulatie_10jaar.xlsx")

# Features en encoding
feature_cols = [
    "Leeftijd", "Geslacht", "Functie", "ContractType", "Dienstjaren", "Werkuren"
] + [f"Verzuimdagen_{y}" for y in range(2014, 2024)] + [f"Verzuimmomenten_{y}" for y in range(2014, 2024)]
df_encoded = pd.get_dummies(df[feature_cols], drop_first=True)

# Targets
y_clf = df["VerzuimVolgendJaar"]
y_reg = df["VerwachteVerzuimdagen"]

# Train/test split
X_train, X_test, y_train_clf, y_test_clf = train_test_split(df_encoded, y_clf, test_size=0.2, random_state=42)
_, _, y_train_reg, y_test_reg = train_test_split(df_encoded, y_reg, test_size=0.2, random_state=42)

# Modellen trainen
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
clf_model.fit(X_train, y_train_clf)
reg_model.fit(X_train, y_train_reg)

# Opslaan
joblib.dump(clf_model, "model_classification.pkl")
joblib.dump(reg_model, "model_regression.pkl")
joblib.dump(df_encoded.columns.tolist(), "model_features.pkl")
