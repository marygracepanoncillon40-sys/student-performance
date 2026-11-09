# train_model.py
import io
import urllib.request
import zipfile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------
# Part 1: Retrieve the dataset (Mathematics file: student-mat.csv)
# ---------------------------------------------------------------
# Note: the UCI "Student Performance" dataset (ID 320) bundles two files,
# student-mat.csv (Mathematics, 395 rows) and student-por.csv (Portuguese,
# 649 rows). Downloading the archive lets us pick the Mathematics file
# explicitly instead of relying on the combined feature/target tables.
URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

with urllib.request.urlopen(URL, timeout=60) as response:
    outer = zipfile.ZipFile(io.BytesIO(response.read()))

# The archive may contain student.zip (nested) or the CSV files directly.
if "student-mat.csv" in outer.namelist():
    df = pd.read_csv(outer.open("student-mat.csv"), sep=";")
else:
    inner_name = next(n for n in outer.namelist() if n.endswith(".zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    df = pd.read_csv(inner.open("student-mat.csv"), sep=";")

df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)  # expected: (395, 33)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())

# ---------------------------------------------------------------
# Part 2: Preprocess, train, and evaluate
# ---------------------------------------------------------------
# Target: 1 = at risk (G3 < 10); 0 = not at risk
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor (target leakage)
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
))

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl")
