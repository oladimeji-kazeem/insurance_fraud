# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# %% [code] Cell 2
df = pd.read_excel('data/insurance_claim2.xlsx')

# %% [code] Cell 3
df.head()

# %% [code] Cell 4
missing = df.isnull().sum()

# %% [code] Cell 5
missing

# %% [code] Cell 6
df.shape

# %% [code] Cell 7
df["Resolution Date"] = pd.to_datetime(df["Resolution Date"], errors="coerce")

# %% [code] Cell 8
df = df.dropna(subset=["Resolution Date"]).copy()

# %% [code] Cell 9
df.isnull().sum()

# %% [code] Cell 10
date_columns = [
    "Claim Date",
    "Resolution Date",
    "Policy Expiration Date"
]


for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# %% [code] Cell 11
numeric_columns = ["Claim Amount (NGN)",
                  "Latitude",
                  "Longitude",
                  "Previous Claims Count",
                  "Policy Tenure (Years)",
                  "Number of Policy Violations",
                  "Documents Submitted",
                  "Coverage Limits (NGN)",
                  "Claim_to_Limit_Ratio",
                  "Days_past_expiration",
                  "Days_Past_Expiration",
                  "Days_Remaining_On_Policy",
                  "Required Documents Count",
                  "Document_Completeness_Ratio",
                  "Exclusion_Clause_Triggered",
                  "Fraud_Risk_Score",
                  "Requires_Senior_Approval",
                  "Exclusion_Risk_Index",
                  "Account_Tier_Level",
                  "Is_Premium_Or_VIP"]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# %% [code] Cell 12
df = df.drop_duplicates()

# %% [markdown] Cell 13
# ### New Fraud Risk Features

# %% [code] Cell 14
df["Invalid_ClaimAmount_Flag"] = (df["Claim Amount (NGN)"] < 0).fillna(False).astype(int)

# %% [code] Cell 15
df["Invalid_CoverageLimit_Flag"] = (df["Coverage Limits (NGN)"] <= 0).fillna(False).astype(int)

# %% [code] Cell 16
df["Claim_Exceeds_Limit_Flag"] = (df["Claim Amount (NGN)"] > df["Coverage Limits (NGN)"]).fillna(False).astype(int)

# %% [code] Cell 17
df["Invalid_Policy_Tenure_Flag"] = (df["Policy Tenure (Years)"] < 0).fillna(False).astype(int)

# %% [code] Cell 18
df["Invalid_Document_Count_Flag"] = ((df["Documents Submitted"] < 0) | (df["Required Documents Count"] < 0)).fillna(False).astype(int)

# %% [code] Cell 19
#Claim amount relative to Ratio
df["Claim_to_Coverage_ratio"] = (df["Claim Amount (NGN)"]/df["Coverage Limits (NGN)"]).replace(0, np.nan)

# %% [code] Cell 20
#Document Completeness calc
required = df["Required Documents Count"].replace(0, np.nan)

df["Document_Completeness_calc"] = (
    df["Documents Submitted"] / required
).clip(0, 1)

df["Document_Incomplete_Flag"] = (
    df["Documents Submitted"] < df["Required Documents Count"]
).fillna(False).astype(int)

# %% [code] Cell 21
# Claim datae relative to policy expiry
df["Days_from_expiry_to_claim"] = (
    df["Claim Date"] - df["Policy Expiration Date"]
).dt.days
    

# %% [markdown] Cell 22
# ### Cleaning the categorical Features

# %% [code] Cell 23
cat_columns = [
    "Product Line",
    "Claim Status",
    "Reason",
    "Insurance Company",
    "Country",
    "State",
    "City",
    "Claim Submission Channel",
    "Communication Preference",
    "Account Type"
]

for col in cat_columns:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

# %% [markdown] Cell 24
# ### Final Data Preprocessing for Model Development

# %% [code] Cell 25
#exclude identifiers and fields that may leak future decisions
exclude_columns = [
    "Claim ID",
    "Customer ID",
    "Resolution Data",
    "Resoluyion",
    "Fraud_Risk_Score",
    "Requires_Senior_Approval",
    "Claim Status",
    "Days_past_expiration",
    "Days_Past_Expiration",
    "Days_Remaining_On_Policy",
    "Claim_to_Limit_Ratio",
    "Document_Completeness_Ratio"
]

if "fraud_label" in df.columns:
    exclude_columns.append("fraud_label")

# %% [code] Cell 26
X = df.drop(columns=[c for c in exclude_columns if c in df.columns])

# %% [code] Cell 27
num_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

# %% [code] Cell 28
cat_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

# %% [code] Cell 29
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# %% [code] Cell 30
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# %% [code] Cell 31
preprocessor_pipeline = ColumnTransformer([
    ("numeric", num_pipeline, num_features),
    ("categorical", cat_pipeline, cat_features)
])

# %% [code] Cell 32
X_processed = preprocessor_pipeline.fit_transform(X)

# %% [code] Cell 33
X.shape

# %% [code] Cell 34
X_processed

# %% [code] Cell 35

### Determining the Fraud label field

# %% [code] Cell 36
conditions = [
    df["Fraud_Risk_Score"].between(0.0000, 0.3999, inclusive="both"),
    df["Fraud_Risk_Score"].between(0.4000, 1.0000, inclusive="both"),
]

categories = ["Low Risk", "High Risk"]

df["Fraud_Risk_Category"] = np.select(conditions, categories, default="Review Score")

# treating the missing fraud scores
df.loc[df["Fraud_Risk_Score"].isna(), "Fraud_Risk_Category"] = "Missing Score"

# %% [code] Cell 37
df.head()

# %% [code] Cell 38
df['Fraud_Risk_Category'].value_counts()

# %% [code] Cell 39
df1 = df.copy()

# %% [code] Cell 40
df['Fraud_Risk_Category'] = df['Fraud_Risk_Category'].map({
    "Low Risk": 0,
    "High Risk": 1
})

# %% [code] Cell 41
y = df['Fraud_Risk_Category']

# %% [code] Cell 42
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.30, random_state=42, stratify=y)

# %% [code] Cell 43
len(X_train)

# %% [code] Cell 44
model = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)

# %% [code] Cell 45
model.fit(X_train, y_train)

# %% [code] Cell 46
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# %% [code] Cell 47
y_pred

# %% [code] Cell 48
y_pred_proba

# %% [code] Cell 49
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %% [code] Cell 50
accuracy = accuracy_score(y_test, y_pred)

# %% [code] Cell 51
accuracy

# %% [code] Cell 52
importances = model.feature_importances_

# %% [code] Cell 53
importances

# %% [code] Cell 54
joblib.dump(model, 'insurance_fraud_detection_model.pkl')
#print("Model Saved")               
