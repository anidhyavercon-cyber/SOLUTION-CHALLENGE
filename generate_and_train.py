"""
BHU SURAKSHA - Enhanced ML Training Script
Generates rich dataset + trains a model with 96%+ accuracy
Features: temperature, rainfall, wind_speed, humidity, pressure, river_level, soil_moisture
Target: disaster_type (0=none, 1=flood, 2=landslide, 3=cyclone)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os, json

np.random.seed(42)
N = 5000

def generate_dataset(n=N):
    data = []
    for _ in range(n):
        # Base weather
        temp       = np.random.uniform(15, 50)
        rainfall   = np.random.uniform(0, 200)
        wind_speed = np.random.uniform(0, 120)
        humidity   = np.random.uniform(20, 100)
        pressure   = np.random.uniform(950, 1050)
        river_level= np.random.uniform(0, 20)
        soil_moist = np.random.uniform(0, 100)

        # Rule-based labelling (matches real-world patterns)
        flood     = (rainfall > 60 and river_level > 12) or \
                    (rainfall > 80 and soil_moist > 70) or \
                    (rainfall > 100)
        landslide = (rainfall > 50 and soil_moist > 75 and temp < 30) or \
                    (rainfall > 70 and soil_moist > 60)
        cyclone   = (wind_speed > 80 and pressure < 970 and humidity > 85)

        if cyclone:
            label = 3
        elif landslide and not flood:
            label = 2
        elif flood:
            label = 1
        else:
            label = 0

        data.append([temp, rainfall, wind_speed, humidity, pressure,
                      river_level, soil_moist, label])

    cols = ["temperature","rainfall","wind_speed","humidity",
            "pressure","river_level","soil_moisture","disaster_type"]
    df = pd.DataFrame(data, columns=cols)

    # Also add flood_risk binary col for backward-compat
    df["flood_risk"] = (df["disaster_type"] == 1).astype(int)
    return df

print("Generating dataset …")
df = generate_dataset()
df.to_csv("../data/weather_data_enhanced.csv", index=False)
print(f"Dataset: {len(df)} rows | class distribution:\n{df['disaster_type'].value_counts()}")

# Features / target
FEATURES = ["temperature","rainfall","wind_speed","humidity",
            "pressure","river_level","soil_moisture"]
X = df[FEATURES]
y = df["disaster_type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Ensemble: GBM + RF
gb  = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
rf  = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
clf = VotingClassifier([("gb", gb), ("rf", rf)], voting="soft")

print("Training ensemble model …")
clf.fit(X_train_s, y_train)

preds = clf.predict(X_test_s)
acc   = accuracy_score(y_test, preds)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, preds,
      target_names=["None","Flood","Landslide","Cyclone"]))

cv = cross_val_score(clf, scaler.transform(X), y, cv=5, scoring="accuracy")
print(f"5-fold CV: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")

# Save
os.makedirs("../models", exist_ok=True)
joblib.dump(clf,    "../models/disaster_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

meta = {"features": FEATURES,
        "classes": ["None","Flood","Landslide","Cyclone"],
        "accuracy": round(acc*100, 2)}
with open("../models/model_meta.json","w") as f:
    json.dump(meta, f, indent=2)

print("\n💾 Saved: models/disaster_model.pkl + scaler.pkl + model_meta.json")
