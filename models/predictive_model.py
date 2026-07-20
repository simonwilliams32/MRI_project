#BASELINE MODEL - logistic regression model for predicting management strategy, 81% baseline accuracy.
import pandas as pd
import numpy as np
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix)
import joblib
from pathlib import Path
import os
from datetime import datetime

#LOAD THE DATASET
def load_data():
    data = pd.read_excel("raw_data/synthetic_vestibular_schwannoma_data.xlsx")
    return data

#BASELINE PREPROC PIPELINE
numeric_cols = ["BMI", "Scan 1: Volume", "linear_regression_growth_rate_cm3", "charlson_comorbidity_index", "age_at_diagnosis"]
categorical_cols = ["Sex", "tumour_laterality"]

def preprocess():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            drop="if_binary",
            handle_unknown="ignore"
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])
    return preprocessor

#Train test split
data = load_data()
feature_cols = numeric_cols + categorical_cols
X = data[feature_cols]
y = data["Management"]
print(f"Data Shape: {data.shape}")
print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")

#preprocessor_test = preprocess()
#X_transform_test = preprocessor_test.fit_transform(X)
#print(f"shape of test: {X_transform_test.shape}")


def create_ensemble_model():
    preprocessor = preprocess()

    #SVM
    svm_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", SVC(
        kernel="rbf",
        C=50,
        gamma=0.01,
        probability=True,
        random_state=42
        ))
    ])

    #Random forest
    RF_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=380,
            max_depth=10,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ))
    ])

    #KNN
    knn_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", KNeighborsClassifier(
            n_neighbors=10
        ))
    ])

    #XGBoost
    XGB_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42
        ))
    ])

    #Logistic Regressor
    log_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            C=0.1
        ))
    ])

    #Creating a soft-voting ensemble model pipeline that votes
    ensemble_model = VotingClassifier(
        estimators=[("logistic_regression", log_pipeline), ("svm", svm_pipeline), ("random_forest", RF_pipeline), ("knn", knn_pipeline), ("xgboost", XGB_pipeline)],
        voting="soft",       #soft rather than hard voting = based on confidence rather than single vote
        weights=None,
        n_jobs=-1
    )
    print("Ensemble Model Established Successfully")

    return ensemble_model

#Train model
def train_model():
    ensemble_model = create_ensemble_model()
    ensemble_model.fit(X, y)
    print("Ensemble Model Trained Successfully")

    now = datetime.now()
    f = now.strftime("%d_%H_%M")
    username = os.environ.get('USER') or os.environ.get('LOGNAME')
    model_name = username+"_"+f
    model_name

    #Saving the trained model
    MODEL_PATH = Path(f"saved_models/{model_name}.joblib")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble_model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    return ensemble_model

test_model = train_model()


#def load_model():
