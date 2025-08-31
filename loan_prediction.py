import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def load_data(path="loan_data.csv"):
    df = pd.read_csv(path)
    return df

def build_pipeline(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])
    clf = Pipeline(steps=[
        ("preproc", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    return clf

def main():
    df = load_data()
    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column as the target.")
    X = df.drop(columns=["label"])
    y = df["label"]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = build_pipeline(numeric_cols, categorical_cols)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))

    joblib.dump(pipeline, "loan_rf_pipeline.joblib")
    print("Saved model to loan_rf_pipeline.joblib")

if __name__ == "__main__":
    main()
