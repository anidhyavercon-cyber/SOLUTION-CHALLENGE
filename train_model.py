"""
BHU SURAKSHA - Multi-Model ML Training with Overfit/Underfit Detection
Weather enrichment via OpenWeatherMap API (real live snapshots).
Run from /training/ folder: python train_model.py
"""
import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from sklearn.linear_model    import LogisticRegression, RidgeClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier,
                                     VotingClassifier, AdaBoostClassifier,
                                     ExtraTreesClassifier)
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (accuracy_score, precision_score,
                                      recall_score, f1_score,
                                      classification_report, roc_auc_score)
import joblib
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[WARN] pip install imbalanced-learn")

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "..", "data", "india_full_weather_dataset.csv")
MODEL_DIR = os.path.join(BASE, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── OpenWeatherMap Config ──────────────────────────────────────────────────
OWM_API_KEY = "73ab8a4dad0f433c9a7e2e6dc2087674"
OWM_BASE    = "https://api.openweathermap.org/data/2.5"

# Key Indian cities for live data enrichment (lat, lon, is_flood_prone)
OWM_CITIES = [
    # Name,                 lat,    lon,   flood_prone
    ("Patna",              25.59,  85.14,  True),
    ("Guwahati",           26.14,  91.74,  True),
    ("Bhubaneswar",        20.27,  85.83,  True),
    ("Kolkata",            22.57,  88.36,  True),
    ("Mumbai",             18.97,  72.82,  True),
    ("Chennai",            13.08,  80.27,  True),
    ("Bengaluru",          12.97,  77.59,  False),
    ("Hyderabad",          17.36,  78.47,  False),
    ("Jaipur",             26.91,  75.79,  False),
    ("Lucknow",            26.84,  80.94,  True),
    ("Dehradun",           30.32,  78.03,  True),
    ("Imphal",             24.80,  93.94,  True),
    ("Shillong",           25.57,  91.88,  True),
    ("Agartala",           23.83,  91.28,  True),
    ("Itanagar",           27.10,  93.62,  True),
    ("Delhi",              28.66,  77.21,  False),
    ("Chandigarh",         30.73,  76.77,  False),
    ("Bhopal",             23.25,  77.41,  False),
    ("Raipur",             21.25,  81.62,  False),
    ("Thiruvananthapuram",  8.52,  76.94,  True),
]

print("=" * 70)
print("  BHU SURAKSHA - Multi-Model Training + Overfit/Underfit Analysis")
print("=" * 70)

# ── Helper: Fetch live weather snapshot from OWM ──────────────────────────
def _fetch_owm_snapshot(lat: float, lon: float) -> dict | None:
    """Fetch current conditions from OpenWeatherMap (metric units)."""
    try:
        url = (f"{OWM_BASE}/weather?lat={lat}&lon={lon}"
               f"&appid={OWM_API_KEY}&units=metric")
        r   = requests.get(url, timeout=6)
        r.raise_for_status()
        d   = r.json()
        m   = d.get("main", {})
        w   = d.get("wind", {})
        rain = d.get("rain", {}).get("1h", 0.0)
        snow = d.get("snow", {}).get("1h", 0.0)
        return {
            "temperature": float(m.get("temp",     25.0)),
            "rainfall":    round(rain + snow, 2),
            "wind_speed":  round(w.get("speed", 0) * 3.6, 1),  # m/s -> km/h
            "humidity":    float(m.get("humidity", 60.0)),
            "pressure":    float(m.get("pressure", 1013.0)),
        }
    except Exception as e:
        return None

# ── 1. Load & Augment ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df.dropna(subset=["temperature","rainfall","wind_speed","flood_risk"], inplace=True)

floods    = df[df.flood_risk == 1].copy()
no_floods = df[df.flood_risk == 0].sample(5000, random_state=42).copy()
combined  = pd.concat([floods, no_floods], ignore_index=True)

combined["month"]    = combined["date"].dt.month
combined["season"]   = combined["month"].map(
    {12:1,1:1,2:1, 3:2,4:2,5:2, 6:3,7:3,8:3, 9:4,10:4,11:4})
combined["humidity"] = np.clip(
    40 + combined["rainfall"]*1.8 - combined["temperature"]*0.3
    + rng.normal(0, 6, len(combined)), 20, 99)
base_p = 1013 - combined["lat"]*0.1 - (combined["season"]==3).astype(int)*8
combined["pressure"]     = np.clip(base_p + rng.normal(0,5,len(combined)), 950, 1050)
combined["river_level"]  = np.clip(combined["rainfall"]*0.4 + rng.normal(0,2,len(combined)), 0, 25)
combined["soil_moisture"]= np.clip(combined["humidity"]*0.5 + combined["rainfall"]*0.8
                                    + rng.normal(0,8,len(combined)), 0, 100)

for col, scale in [("temperature",3.0),("rainfall",8.0),("wind_speed",5.0)]:
    combined[col] = np.clip(combined[col] + rng.normal(0, scale, len(combined)), 0, None)

combined["month_sin"] = np.sin(2*np.pi*combined["month"]/12)
combined["month_cos"] = np.cos(2*np.pi*combined["month"]/12)
combined["rain_wind"] = combined["rainfall"] * combined["wind_speed"]

def stochastic_label(row):
    p = 0.0
    if row["rainfall"]    > 25: p += 0.50
    if row["rainfall"]    > 40: p += 0.25
    if row["river_level"] > 10: p += 0.20
    if row["soil_moisture"]>70: p += 0.10
    p = min(p, 0.95)
    return int(rng.random() < (0.6*row["flood_risk"] + 0.4*p))

combined["label"] = combined.apply(stochastic_label, axis=1)

print(f"\n[CSV base] Dataset: {len(combined):,} rows | "
      f"Floods={combined['label'].sum()} | Non-flood={(combined['label']==0).sum():,}")

# ── 2B. Enrich with Live OWM Weather Snapshots ────────────────────────────
print("\n[2B] Fetching live OWM weather for Indian cities ...")
month_now = datetime.utcnow().month
season_now = {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}.get(month_now, 3)
live_rows  = []
failed     = 0

for city_name, lat, lon, flood_prone in OWM_CITIES:
    snap = _fetch_owm_snapshot(lat, lon)
    if snap is None:
        failed += 1
        continue

    rain  = snap["rainfall"]
    hum   = snap["humidity"]
    wind  = snap["wind_speed"]
    temp  = snap["temperature"]
    pres  = snap["pressure"]

    # Derive dependent features
    river  = max(0.0, rain * 0.4 + rng.normal(0, 0.5))
    soil   = min(100.0, hum * 0.5 + rain * 0.8 + float(rng.normal(0, 3)))
    m_sin  = float(np.sin(2 * np.pi * month_now / 12))
    m_cos  = float(np.cos(2 * np.pi * month_now / 12))

    # Stochastic label based on current conditions
    p = 0.0
    if rain > 25:        p += 0.50
    if rain > 40:        p += 0.25
    if river > 10:       p += 0.20
    if soil > 70:        p += 0.10
    if wind > 60:        p += 0.10
    p = min(p, 0.95)
    base_flood = 1 if flood_prone and rain > 15 else 0
    label = int(rng.random() < (0.6 * base_flood + 0.4 * p))

    live_rows.append({
        "temperature":  temp,
        "rainfall":     rain,
        "wind_speed":   wind,
        "humidity":     hum,
        "pressure":     pres,
        "river_level":  river,
        "soil_moisture":soil,
        "lat":          lat,
        "lon":          lon,
        "month_sin":    m_sin,
        "month_cos":    m_cos,
        "season":       season_now,
        "rain_wind":    rain * wind,
        "label":        label,
        "flood_risk":   base_flood,
    })
    print(f"    {city_name:<24}  T={temp}C  Rain={rain}mm  "
          f"Wind={wind}km/h  Label={'FLOOD' if label else 'safe'}")

if live_rows:
    live_df = pd.DataFrame(live_rows)
    # Replicate live rows ~30x so they have meaningful weight vs 5k CSV rows
    live_df = pd.concat([live_df] * 30, ignore_index=True)
    # Small noise so duplicates aren't identical
    for col in ["temperature","rainfall","wind_speed","humidity","pressure"]:
        live_df[col] = np.clip(
            live_df[col] + rng.normal(0, 0.5, len(live_df)), 0, None)
    live_df["rain_wind"] = live_df["rainfall"] * live_df["wind_speed"]
    combined = pd.concat([combined, live_df], ignore_index=True)
    print(f"    OWM enrichment: +{len(live_df):,} rows from {len(live_rows)//30} cities "
          f"(failed={failed})")
else:
    print(f"    OWM enrichment skipped (all {len(OWM_CITIES)} cities failed).")

FEATURES = ["temperature","rainfall","wind_speed","humidity","pressure",
            "river_level","soil_moisture","lat","lon",
            "month_sin","month_cos","season","rain_wind"]
X = combined[FEATURES].values
y = combined["label"].values

print(f"\nDataset: {len(combined):,} rows | Floods={y.sum()} | Non-flood={(y==0).sum():,}")

# ── 2. Split ───────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

# ── 3. SMOTE ───────────────────────────────────────────────────────────────
if HAS_SMOTE and y_train.sum() >= 3:
    k = min(5, y_train.sum()-1)
    sm = SMOTE(random_state=42, k_neighbors=k)
    X_tr, y_tr = sm.fit_resample(X_train, y_train)
    print(f"SMOTE: {len(X_tr):,} balanced samples")
else:
    X_tr, y_tr = X_train, y_train

scaler    = StandardScaler()
X_tr_s    = scaler.fit_transform(X_tr)
X_test_s  = scaler.transform(X_test)
X_train_s = scaler.transform(X_train)   # unaugmented, for bias-variance

# ── 4. Model Zoo (10 models) ───────────────────────────────────────────────
candidates = [
    ("Logistic Regression",
     LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)),

    ("Ridge Classifier",
     RidgeClassifier(class_weight="balanced")),

    ("K-Nearest Neighbours",
     KNeighborsClassifier(n_neighbors=7)),

    ("Naive Bayes",
     GaussianNB()),

    ("Decision Tree",
     DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42)),

    ("AdaBoost",
     AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)),

    ("Extra Trees",
     ExtraTreesClassifier(n_estimators=150, class_weight="balanced",
                          random_state=42, n_jobs=-1)),

    ("Random Forest",
     RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                            random_state=42, n_jobs=-1)),

    ("Gradient Boosting",
     GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                 learning_rate=0.08, subsample=0.8, random_state=42)),

    ("Voting Ensemble (GBM+RF+ET)",
     VotingClassifier([
         ("gb", GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                           learning_rate=0.08, random_state=42)),
         ("rf", RandomForestClassifier(n_estimators=150, max_depth=10,
                                        class_weight="balanced", random_state=42, n_jobs=-1)),
         ("et", ExtraTreesClassifier(n_estimators=150, class_weight="balanced",
                                      random_state=42, n_jobs=-1)),
     ], voting="soft")),
]

# ── 5. Train, Evaluate, Bias-Variance Check ────────────────────────────────
print("\n" + "="*70)
print(f"  {'MODEL':<30} {'TrainAcc':>9} {'TestAcc':>9} {'F1':>7} {'AUC':>7} {'STATUS':<15}")
print("="*70)

results = []
for name, clf in candidates:
    t0 = time.time()
    clf.fit(X_tr_s, y_tr)

    # Train score on ORIGINAL (unaugmented) training set
    tr_pred  = clf.predict(X_train_s)
    tr_acc   = accuracy_score(y_train, tr_pred)

    # Test scores
    te_pred  = clf.predict(X_test_s)
    te_acc   = accuracy_score(y_test,  te_pred)
    prec     = precision_score(y_test, te_pred, zero_division=0)
    rec      = recall_score(y_test,    te_pred, zero_division=0)
    f1       = f1_score(y_test,        te_pred, zero_division=0)
    proba    = (clf.predict_proba(X_test_s)[:,1]
                if hasattr(clf,"predict_proba") else None)
    auc      = roc_auc_score(y_test, proba) if proba is not None else 0.0

    # Diagnostic using F1-score (more robust for imbalanced data)
    f1_train = f1_score(y_train, tr_pred, zero_division=0)
    gap_f1   = f1_train - f1
    gap      = tr_acc - te_acc
    
    if f1 < 0.30:
        status = "UNDERFIT"
    elif gap_f1 > 0.15:
        status = "OVERFIT"
    else:
        status = "GOOD FIT"

    elapsed = time.time() - t0
    print(f"  {name:<30} {tr_acc*100:>8.2f}% {te_acc*100:>8.2f}%"
          f" {f1*100:>6.2f}% {auc*100:>6.2f}%  {status}")

    results.append({
        "name":       name,
        "train_acc":  round(tr_acc*100, 2),
        "accuracy":   round(te_acc*100, 2),
        "precision":  round(prec*100, 2),
        "recall":     round(rec*100, 2),
        "f1":         round(f1*100,  2),
        "f1_train":   f1_train,
        "gap_f1":     gap_f1,
        "auc_roc":    round(auc*100, 2),
        "gap":        round(gap*100, 2),
        "status":     status,
        "train_sec":  round(elapsed, 2),
        "model":      clf,
    })

# ── 6. Rank: prefer GOOD FIT; among those, highest F1 then AUC ───────────
def rank_key(r):
    fit_score = 2 if r["status"]=="GOOD FIT" else (1 if r["status"]=="OVERFIT" else 0)
    return (fit_score, r["f1"], r["auc_roc"])

results.sort(key=rank_key, reverse=True)
best = results[0]

print("\n" + "="*70)
print(f"  WINNER: {best['name']}")
print(f"  Status:    {best['status']}")
print(f"  Train Acc: {best['train_acc']}%  |  Test Acc: {best['accuracy']}%  |  Gap: {best['gap']}%")
print(f"  F1: {best['f1']}%  |  AUC-ROC: {best['auc_roc']}%")
print("="*70)

print("\nFull Classification Report (best model):")
best_pred = best["model"].predict(X_test_s)
print(classification_report(y_test, best_pred,
      target_names=["No Flood","Flood"], zero_division=0))

# ── 7. Save ────────────────────────────────────────────────────────────────
joblib.dump(best["model"], os.path.join(MODEL_DIR, "disaster_model.pkl"))
joblib.dump(scaler,         os.path.join(MODEL_DIR, "scaler.pkl"))

meta = {
    "features":       FEATURES,
    "classes":        ["No Flood", "Flood"],
    "best_model":     best["name"],
    "accuracy":       best["accuracy"],
    "train_accuracy": best["train_acc"],
    "precision":      best["precision"],
    "recall":         best["recall"],
    "f1":             best["f1"],
    "auc_roc":        best["auc_roc"],
    "overfit_gap":    best["gap"],
    "fit_status":     best["status"],
    "n_samples":      len(combined),
    "smote_applied":  HAS_SMOTE,
    "dataset_source": "india_full_weather_dataset.csv + noise augmentation",
    "trained_at":     datetime.utcnow().isoformat(),
}
with open(os.path.join(MODEL_DIR,"model_meta.json"),"w") as f:
    json.dump(meta, f, indent=2)

# Leaderboard (no model objects)
leaderboard = [{k:v for k,v in r.items() if k!="model"} for r in results]
with open(os.path.join(MODEL_DIR,"model_comparison.json"),"w") as f:
    json.dump(leaderboard, f, indent=2)

# Detailed Overfit Report for UI
overfit_report = []
for r in results:
    sugg = ""
    if r["status"] == "OVERFIT":
        sugg = "Reduce max_depth, increase min_samples_leaf, or add more real-world disaster samples."
    elif r["status"] == "UNDERFIT":
        sugg = "Increase model complexity or features. Try more boosting rounds."
    else:
        sugg = "Model is well-balanced. Monitor for concept drift."
        
    overfit_report.append({
        "model": r["name"],
        "status": r["status"],
        "note": f"Train F1={r.get('f1_train',0)*100:.1f}% vs Test F1={r['f1']:.1f}%",
        "train_f1": round(r.get("f1_train", 0)*100, 2),
        "test_f1": r["f1"],
        "gap": round(r.get("gap_f1", 0)*100, 2),
        "suggestion": sugg
    })
with open(os.path.join(MODEL_DIR, "overfit_report.json"), "w") as f:
    json.dump(overfit_report, f, indent=2)

print("\nSaved: disaster_model.pkl | scaler.pkl | model_meta.json | model_comparison.json")
print(f"Best: {best['name']} | Fit: {best['status']} | F1: {best['f1']}% | AUC: {best['auc_roc']}%")
