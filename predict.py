"""
Hospital Readmission Risk Predictor
------------------------------------
Predicts 30-day readmission risk for diabetic patients.
Explains predictions via Monte Carlo Shapley attribution.

Dataset: Diabetes 130-US hospitals (1999-2008)
Usage:   python predict.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from shapley import mc_shapley, display_shapley
import warnings
warnings.filterwarnings("ignore")


# ── 1. LOAD & CLEAN ──────────────────────────────────────────────────────────

def load_data(path="diabetic_data.csv"):
    df = pd.read_csv(path)
    df.replace("?", np.nan, inplace=True)
    df.drop(columns=["weight", "payer_code", "medical_specialty"], inplace=True)
    df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)
    df.drop(columns=["readmitted", "encounter_id", "patient_nbr"], inplace=True)
    return df


def preprocess(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("missing")
        df[col] = le.fit_transform(df[col].astype(str))
    df.fillna(df.median(), inplace=True)
    return df


# ── 2. TRAIN ─────────────────────────────────────────────────────────────────

def train_model(df):
    X = df.drop(columns=["readmitted_30"])
    y = df["readmitted_30"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("\n── Model Evaluation ──────────────────────────")
    print(classification_report(y_test, model.predict(X_test),
                                target_names=["Not Readmitted", "Readmitted <30d"]))

    # Return training data as numpy — used as background distribution in Shapley
    return model, X.columns.tolist(), X_train.values


# ── 3. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df = load_data("diabetic_data.csv")
    df = preprocess(df)

    print("Training model (~30 seconds)...")
    model, feature_names, background = train_model(df)

    # Demo patient — real row from dataset
    patient = df.drop(columns=["readmitted_30"]).iloc[42].values
    patient_prob = model.predict_proba(patient.reshape(1, -1))[0][1]

    print("Computing Shapley values (~20 seconds)...")
    shapley_vals = mc_shapley(model, feature_names, patient, background, n_samples=512)

    display_shapley(patient_prob, shapley_vals)
