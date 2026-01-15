

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def load_models():
    
    df = pd.read_csv("space_missions_dataset.csv")

    encoders = {}

   
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    
    feature_cols = [
        col for col in df.columns
        if col not in [
            "Mission ID",
            "Mission Name",
            "Launch Date",
            "Mission Type",
            "Mission Cost (billion USD)",
            "Mission Success (%)"
        ]
    ]

    X = df[feature_cols]

    
    y_mission = df["Mission Type"]

    X_train, _, y_train, _ = train_test_split(
        X, y_mission, test_size=0.2, random_state=42
    )

    mission_model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )
    mission_model.fit(X_train, y_train)

  
    y_cost = df["Mission Cost (billion USD)"]

    X_train, _, y_train, _ = train_test_split(
        X, y_cost, test_size=0.2, random_state=42
    )

    cost_model = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )
    cost_model.fit(X_train, y_train)

   
    y_success = df["Mission Success (%)"]

    X_train, _, y_train, _ = train_test_split(
        X, y_success, test_size=0.2, random_state=42
    )

    success_model = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )
    success_model.fit(X_train, y_train)

    return {
        "mission_model": mission_model,
        "cost_model": cost_model,
        "success_model": success_model,
        "encoders": encoders,
        "features": feature_cols
    }
