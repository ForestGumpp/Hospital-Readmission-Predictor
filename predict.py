import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from shapley import mc_shapley, display_shapley
import warnings
warnings.filterwarnings("ignore")

# predicts whether a diabetic patient will be readmitted within 30 days
# dataset: diabetes 130-US hospitals (1999-2008)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "diabetic_data.csv"))

df.replace("?", np.nan, inplace=True)

# these had way too many missing values / weren't useful
df.drop(columns=["weight", "payer_code", "medical_specialty"], inplace=True)

# turn the target into binary - we only care about <30 day readmits
df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)
df.drop(columns=["readmitted", "encounter_id", "patient_nbr"], inplace=True)

# encode all the categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("missing")
    df[col] = le.fit_transform(df[col].astype(str))

df.fillna(df.median(), inplace=True)

X = df.drop(columns=["readmitted_30"])
y = df["readmitted_30"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("training model...")
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test),
                            target_names=["Not Readmitted", "Readmitted <30d"]))

feature_names = X.columns.tolist()
background = X_train.values

# pick a random patient from the test set to explain
patient = X.iloc[42].values
patient_prob = model.predict_proba(patient.reshape(1, -1))[0][1]

print("computing shapley values, this takes a bit...")
shapley_vals = mc_shapley(model, feature_names, patient, background, n_samples=512)
display_shapley(patient_prob, shapley_vals)
