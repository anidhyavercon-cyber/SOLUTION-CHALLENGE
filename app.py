"""
BHU SURAKSHA - Backend API v5.0
================================
Weather Sources (priority order):
  1. OpenWeatherMap API  (primary  - real-time, key-based)
  2. Open-Meteo API      (fallback - free, no key needed)
  3. Seasonal defaults   (last resort)

Endpoints:
  GET  /api/health
  POST /api/predict              -- predict from JSON features
  GET  /api/weather/live         -- live weather + ML prediction (OWM primary)
  GET  /api/weather/all-states   -- bulk live weather for all 28+ states
  GET  /api/weather/owm-test     -- test OWM API key connectivity
  GET  /api/models/comparison    -- model comparison leaderboard
  GET  /api/models/overfit       -- overfitting / underfitting report
  GET  /api/zones                -- Uttarakhand risk zones
  GET  /api/alerts               -- live alerts for all zones
  GET  /api/history              -- 30-day history
  GET  /api/model/info           -- model metadata

Run: python app.py  (from /backend/ folder)
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib, json, os, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests

app = Flask(__name__)
CORS(app)

# ─── OpenWeatherMap Config ────────────────────────────────────────────────────
# Free tier: https://openweathermap.org/api
OWM_API_KEY = "73ab8a4dad0f433c9a7e2e6dc2087674"
OWM_BASE    = "https://api.openweathermap.org/data/2.5"

# ─── Load Model ───────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "..", "models")

model  = joblib.load(os.path.join(MODEL_DIR, "disaster_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
    meta = json.load(f)

FEATURES = meta["features"]
CLASSES  = meta["classes"]          # ["No Flood", "Flood"]

# ─── India State Capital Coordinates (for Open-Meteo) ────────────────────────
STATE_COORDS = {
    "Andhra Pradesh":    {"lat": 13.60,  "lon": 79.40,  "capital": "Amaravati"},
    "Arunachal Pradesh": {"lat": 27.10,  "lon": 93.62,  "capital": "Itanagar"},
    "Assam":             {"lat": 26.14,  "lon": 91.74,  "capital": "Guwahati"},
    "Bihar":             {"lat": 25.59,  "lon": 85.14,  "capital": "Patna"},
    "Chhattisgarh":      {"lat": 21.25,  "lon": 81.62,  "capital": "Raipur"},
    "Goa":               {"lat": 15.49,  "lon": 73.82,  "capital": "Panaji"},
    "Gujarat":           {"lat": 23.02,  "lon": 72.57,  "capital": "Gandhinagar"},
    "Haryana":           {"lat": 29.95,  "lon": 76.82,  "capital": "Chandigarh"},
    "Himachal Pradesh":  {"lat": 31.10,  "lon": 77.17,  "capital": "Shimla"},
    "Jharkhand":         {"lat": 23.36,  "lon": 85.33,  "capital": "Ranchi"},
    "Karnataka":         {"lat": 12.97,  "lon": 77.59,  "capital": "Bengaluru"},
    "Kerala":            {"lat": 8.52,   "lon": 76.94,  "capital": "Thiruvananthapuram"},
    "Madhya Pradesh":    {"lat": 23.25,  "lon": 77.41,  "capital": "Bhopal"},
    "Maharashtra":       {"lat": 18.97,  "lon": 72.82,  "capital": "Mumbai"},
    "Manipur":           {"lat": 24.80,  "lon": 93.94,  "capital": "Imphal"},
    "Meghalaya":         {"lat": 25.57,  "lon": 91.88,  "capital": "Shillong"},
    "Mizoram":           {"lat": 23.72,  "lon": 92.72,  "capital": "Aizawl"},
    "Nagaland":          {"lat": 25.67,  "lon": 94.11,  "capital": "Kohima"},
    "Odisha":            {"lat": 20.27,  "lon": 85.83,  "capital": "Bhubaneswar"},
    "Punjab":            {"lat": 30.73,  "lon": 76.77,  "capital": "Chandigarh"},
    "Rajasthan":         {"lat": 26.91,  "lon": 75.79,  "capital": "Jaipur"},
    "Sikkim":            {"lat": 27.33,  "lon": 88.62,  "capital": "Gangtok"},
    "Tamil Nadu":        {"lat": 13.08,  "lon": 80.27,  "capital": "Chennai"},
    "Telangana":         {"lat": 17.36,  "lon": 78.47,  "capital": "Hyderabad"},
    "Tripura":           {"lat": 23.83,  "lon": 91.28,  "capital": "Agartala"},
    "Uttar Pradesh":     {"lat": 26.84,  "lon": 80.94,  "capital": "Lucknow"},
    "Uttarakhand":       {"lat": 30.32,  "lon": 78.03,  "capital": "Dehradun"},
    "West Bengal":       {"lat": 22.57,  "lon": 88.36,  "capital": "Kolkata"},
    "Delhi":             {"lat": 28.66,  "lon": 77.21,  "capital": "New Delhi"},
    "Jammu & Kashmir":   {"lat": 34.08,  "lon": 74.80,  "capital": "Srinagar"},
    "Ladakh":            {"lat": 34.23,  "lon": 77.60,  "capital": "Leh"},
}

# ─── Risk Zones (Uttarakhand) ─────────────────────────────────────────────────
ZONES = [
    {"id":"Z1","name":"Haridwar - Ganga Floodplain","lat":29.9457,"lng":78.1642,
     "base_risk":"flood","no_construction":True,
     "reason":"Low-lying Ganga floodplain, extreme flood history"},
    {"id":"Z2","name":"Rishikesh Valley","lat":30.0869,"lng":78.2676,
     "base_risk":"flood","no_construction":False,
     "reason":"River valley, moderate flood & flash-flood risk"},
    {"id":"Z3","name":"Kedarnath Region","lat":30.7352,"lng":79.0669,
     "base_risk":"landslide","no_construction":True,
     "reason":"High-altitude glacial zone, severe landslide & flash flood"},
    {"id":"Z4","name":"Uttarkashi","lat":30.7268,"lng":78.4354,
     "base_risk":"landslide","no_construction":True,
     "reason":"Seismically active mountain zone, landslide-prone"},
    {"id":"Z5","name":"Dehradun City Core","lat":30.3165,"lng":78.0322,
     "base_risk":"none","no_construction":False,
     "reason":"Relatively stable, monitor during heavy monsoon"},
    {"id":"Z6","name":"Chamoli - Nanda Devi","lat":30.4093,"lng":79.3292,
     "base_risk":"landslide","no_construction":True,
     "reason":"Glacier-melt flood & landslide zone (2021 disaster site)"},
    {"id":"Z7","name":"Pithoragarh Hills","lat":29.5827,"lng":80.2177,
     "base_risk":"landslide","no_construction":False,
     "reason":"Hilly terrain, monsoon landslide risk"},
    {"id":"Z8","name":"Tehri Dam Reservoir","lat":30.3799,"lng":78.4804,
     "base_risk":"flood","no_construction":True,
     "reason":"Dam-breach risk zone, habitation restricted"},
]

PRECAUTIONS = {
    "Flood": [
        "Evacuate low-lying areas immediately",
        "Move to higher ground above the flood line",
        "Avoid crossing flooded roads or bridges",
        "Keep emergency kit: water, food, medicines, documents",
        "Turn off electricity at the mains before evacuation",
        "Follow SDRF / NDRF evacuation routes",
    ],
    "No Flood": [
        "Monitor IMD weather forecasts daily",
        "Keep emergency contacts handy (NDMA: 1078)",
        "Maintain drainage around your property",
        "Participate in local disaster drill programs",
    ],
    "Landslide (Advisory)": [
        "Move away from hillsides and valley channels",
        "Watch for unusual sounds like cracking or rumbling",
        "Stay alert after heavy rainfall for 24-48 hours",
        "Do not construct on steep slopes > 45 degrees",
    ],
    "Cyclone (Advisory)": [
        "Seek strong permanent shelter",
        "Secure loose objects or bring indoors",
        "Stay away from coastal areas",
        "Follow cyclone warning levels (1-5) from IMD",
    ],
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _build_feature_vector(feat: dict) -> list:
    """Build feature vector matching training order."""
    rain    = feat.get("rainfall", 0)
    wind    = feat.get("wind_speed", 0)
    temp    = feat.get("temperature", 25)
    hum     = feat.get("humidity", 60)
    pres    = feat.get("pressure", 1013)
    lat     = feat.get("lat", 20)
    lon     = feat.get("lon", 78)
    month   = feat.get("month", datetime.utcnow().month)
    season  = {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}.get(month, 3)
    river   = feat.get("river_level",  rain * 0.4)
    soil    = feat.get("soil_moisture", min(hum * 0.5 + rain * 0.8, 100))

    import math
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    rain_int  = (0 if rain==0 else 1 if rain<5 else 2 if rain<20 else 3 if rain<60 else 4)
    rain_wind = rain * wind

    return [temp, rain, wind, hum, pres, river, soil,
            lat, lon, month_sin, month_cos, season, rain_wind]


def _run_model(feat_dict: dict):
    vec  = _build_feature_vector(feat_dict)
    arr  = scaler.transform([vec])
    pred = int(model.predict(arr)[0])
    prob = model.predict_proba(arr)[0].tolist()
    return pred, prob


def _rule_advisories(feat: dict) -> list:
    """Simple rule-based advisory (landslide / cyclone)."""
    advisories = []
    if feat.get("rainfall", 0) > 50 and feat.get("wind_speed", 0) > 70:
        advisories.append("Cyclone")
    if feat.get("rainfall", 0) > 40 and feat.get("soil_moisture", 0) > 75:
        advisories.append("Landslide")
    return advisories


# ─── Weather Fetch Functions ──────────────────────────────────────────────────

def _fetch_owm(lat: float, lon: float) -> dict | None:
    """
    PRIMARY: Fetch current weather from OpenWeatherMap API.
    Free tier endpoint: /weather (current conditions).
    Returns: temperature(C), rainfall(mm/h), wind_speed(km/h),
             humidity(%), pressure(hPa), description, city_name.
    """
    url = (
        f"{OWM_BASE}/weather"
        f"?lat={lat}&lon={lon}"
        f"&appid={OWM_API_KEY}"
        "&units=metric"          # Celsius, m/s
    )
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        d    = r.json()
        main = d.get("main", {})
        wind = d.get("wind", {})
        rain = d.get("rain", {}).get("1h", 0)   # mm in last 1 hour
        snow = d.get("snow", {}).get("1h", 0)
        desc = d.get("weather", [{}])[0].get("description", "")
        city = d.get("name", "")

        # OWM wind is in m/s → convert to km/h for consistency
        wind_kmh = round(wind.get("speed", 0) * 3.6, 1)

        return {
            "temperature":   round(main.get("temp",      25.0), 1),
            "feels_like":    round(main.get("feels_like", 25.0), 1),
            "rainfall":      round(rain + snow, 1),
            "wind_speed":    wind_kmh,
            "wind_gust":     round(wind.get("gust", 0) * 3.6, 1),
            "humidity":      round(main.get("humidity",  60.0), 1),
            "pressure":      round(main.get("pressure", 1013.0), 1),
            "visibility_km": round(d.get("visibility", 10000) / 1000, 1),
            "cloud_pct":     d.get("clouds", {}).get("all", 0),
            "description":   desc.title(),
            "city_name":     city,
            "source":        "OpenWeatherMap",
        }
    except Exception as e:
        print(f"[OWM error] {e}")
        return None


def _fetch_open_meteo(lat: float, lon: float) -> dict | None:
    """
    FALLBACK: Fetch current weather from Open-Meteo (free, no key needed).
    Used when OWM is unavailable or rate-limited.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,precipitation,windspeed_10m,"
        "relativehumidity_2m,surface_pressure"
        "&forecast_days=1"
    )
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        c = r.json().get("current", {})
        return {
            "temperature": round(c.get("temperature_2m", 25), 1),
            "rainfall":    round(c.get("precipitation",  0),  1),
            "wind_speed":  round(c.get("windspeed_10m",  10), 1),
            "humidity":    round(c.get("relativehumidity_2m", 60), 1),
            "pressure":    round(c.get("surface_pressure", 1013), 1),
            "source":      "Open-Meteo",
        }
    except Exception as e:
        print(f"[Open-Meteo error] {e}")
        return None


def _fetch_weather(lat: float, lon: float) -> tuple[dict, str]:
    """
    Smart weather fetcher with automatic fallback chain:
      OWM (primary)  -->  Open-Meteo (fallback)  -->  Defaults (last resort)
    Returns (weather_dict, source_label).
    """
    # 1) Try OpenWeatherMap
    w = _fetch_owm(lat, lon)
    if w:
        return w, "OpenWeatherMap (live)"

    # 2) Fallback to Open-Meteo
    w = _fetch_open_meteo(lat, lon)
    if w:
        return w, "Open-Meteo (fallback)"

    # 3) Last resort: seasonal climatological defaults
    month = datetime.utcnow().month
    season_rain = {1:5, 2:5, 3:8, 4:10, 5:15, 6:60, 7:90, 8:80,
                   9:40, 10:20, 11:8, 12:5}.get(month, 10)
    return {
        "temperature": 28.0,
        "rainfall":    float(season_rain),
        "wind_speed":  12.0,
        "humidity":    65.0,
        "pressure":    1010.0,
        "source":      "defaults",
    }, "Seasonal Defaults"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status":        "ok",
        "best_model":    meta.get("best_model", "Ensemble"),
        "model_accuracy":meta.get("accuracy"),
        "train_accuracy":meta.get("train_accuracy"),
        "f1":            meta.get("f1"),
        "train_f1":      meta.get("train_f1"),
        "auc_roc":       meta.get("auc_roc"),
        "cv_f1_mean":    meta.get("cv_f1_mean"),
        "cv_f1_std":     meta.get("cv_f1_std"),
        "fit_status":    meta.get("fit_status"),
        "fit_gap":       meta.get("fit_gap"),
        "version":       "4.0",
        "timestamp":     datetime.utcnow().isoformat()
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    pred, proba = _run_model(data)
    disaster    = CLASSES[pred]
    confidence  = round(max(proba) * 100, 1)
    advisories  = _rule_advisories(data)

    risk_level = "LOW"
    if pred == 1:
        risk_level = "HIGH" if confidence > 70 else "MODERATE"
    elif advisories:
        risk_level = "MODERATE"

    return jsonify({
        "disaster_type":   disaster,
        "risk_level":      risk_level,
        "confidence":      confidence,
        "probabilities":   {CLASSES[i]: round(p*100,1) for i,p in enumerate(proba)},
        "precautions":     PRECAUTIONS.get(disaster, []),
        "advisories":      advisories,
        "model":           meta.get("best_model", "Ensemble"),
        "model_accuracy":  meta.get("accuracy"),
        "timestamp":       datetime.utcnow().isoformat()
    })


@app.route("/api/weather/live")
def weather_live():
    """
    GET /api/weather/live?state=Kerala
    Fetches real-time weather via OWM (primary) → Open-Meteo (fallback),
    then runs the ML model to predict flood/disaster risk.
    """
    state  = request.args.get("state", "Delhi")
    coords = STATE_COORDS.get(state)
    if not coords:
        return jsonify({"error": f"Unknown state: {state}",
                        "valid_states": list(STATE_COORDS.keys())}), 400

    lat, lon          = coords["lat"], coords["lon"]
    weather, src_label = _fetch_weather(lat, lon)

    feat        = {**weather, "lat": lat, "lon": lon,
                   "month": datetime.utcnow().month}
    pred, proba = _run_model(feat)
    disaster    = CLASSES[pred]
    advisories  = _rule_advisories(feat)
    confidence  = round(max(proba) * 100, 1)

    risk_level = (
        "HIGH"     if pred == 1 and confidence > 70 else
        "MODERATE" if pred == 1 or advisories     else
        "LOW"
    )

    return jsonify({
        "state":        state,
        "capital":      coords["capital"],
        "lat":          lat,
        "lon":          lon,
        "weather":      weather,
        "weather_source": src_label,
        "prediction": {
            "disaster_type": disaster,
            "risk_level":    risk_level,
            "confidence":    confidence,
            "probabilities": {CLASSES[i]: round(p*100,1) for i,p in enumerate(proba)},
            "advisories":    advisories,
            "precautions":   PRECAUTIONS.get(disaster, []),
        },
        "model":     meta.get("best_model"),
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/api/weather/all-states")
def weather_all_states():
    """
    GET /api/weather/all-states
    Fetch live OWM weather + ML prediction for ALL Indian states in one call.
    Optional query params:
      ?limit=10   (fetch only first N states, default=all)
      ?risk=high  (filter: high | moderate | low)
    """
    limit      = int(request.args.get("limit", len(STATE_COORDS)))
    risk_filter= request.args.get("risk", "").upper()

    results    = []
    states     = list(STATE_COORDS.items())[:limit]
    now_month  = datetime.utcnow().month

    for state_name, coords in states:
        lat, lon           = coords["lat"], coords["lon"]
        weather, src_label = _fetch_weather(lat, lon)

        feat        = {**weather, "lat": lat, "lon": lon, "month": now_month}
        pred, proba = _run_model(feat)
        disaster    = CLASSES[pred]
        advisories  = _rule_advisories(feat)
        confidence  = round(max(proba) * 100, 1)
        risk_level  = (
            "HIGH"     if pred == 1 and confidence > 70 else
            "MODERATE" if pred == 1 or advisories     else
            "LOW"
        )

        if risk_filter and risk_level != risk_filter:
            continue

        results.append({
            "state":          state_name,
            "capital":        coords["capital"],
            "lat":            lat,
            "lon":            lon,
            "weather":        weather,
            "weather_source": src_label,
            "disaster_type":  disaster,
            "risk_level":     risk_level,
            "confidence":     confidence,
            "advisories":     advisories,
        })

    # Sort by risk level (HIGH first)
    order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    results.sort(key=lambda x: order.get(x["risk_level"], 3))

    high_count = sum(1 for r in results if r["risk_level"] == "HIGH")
    mod_count  = sum(1 for r in results if r["risk_level"] == "MODERATE")

    return jsonify({
        "total_states":  len(results),
        "high_risk":     high_count,
        "moderate_risk": mod_count,
        "low_risk":      len(results) - high_count - mod_count,
        "model":         meta.get("best_model"),
        "timestamp":     datetime.utcnow().isoformat(),
        "states":        results,
    })


@app.route("/api/weather/owm-test")
def owm_test():
    """
    GET /api/weather/owm-test
    Quick connectivity test for the OpenWeatherMap API key.
    Tests against Delhi coordinates.
    """
    lat, lon = 28.66, 77.21   # Delhi
    result   = _fetch_owm(lat, lon)
    if result:
        return jsonify({
            "status":      "ok",
            "message":     "OpenWeatherMap API key is valid and working.",
            "api_key":     OWM_API_KEY[:8] + "..." + OWM_API_KEY[-4:],
            "test_city":   result.get("city_name", "Delhi"),
            "sample_data": result,
            "timestamp":   datetime.utcnow().isoformat(),
        })
    else:
        return jsonify({
            "status":    "error",
            "message":   "OpenWeatherMap API call failed. Check key or network.",
            "api_key":   OWM_API_KEY[:8] + "..." + OWM_API_KEY[-4:],
            "timestamp": datetime.utcnow().isoformat(),
        }), 503


@app.route("/api/models/comparison")
def models_comparison():
    """Return leaderboard of all trained models."""
    cmp_path = os.path.join(MODEL_DIR, "model_comparison.json")
    if not os.path.exists(cmp_path):
        return jsonify({"error": "model_comparison.json not found"}), 404
    with open(cmp_path) as f:
        data = json.load(f)
    return jsonify({
        "best_model":    meta.get("best_model"),
        "models":        data,
        "metrics_note": ("Ranked by F1 on held-out test set (20%). "
                          "Heavily overfitting models (gap>15%) penalised. "
                          "Cross-validation (5-fold) used for robustness.")
    })


@app.route("/api/models/overfit")
def models_overfit():
    """Return overfitting/underfitting diagnostic report."""
    report_path = os.path.join(MODEL_DIR, "overfit_report.json")
    if not os.path.exists(report_path):
        return jsonify({"error": "overfit_report.json not found – retrain first"}), 404
    with open(report_path) as f:
        report = json.load(f)
    return jsonify({
        "best_model":        meta.get("best_model"),
        "best_fit_status":   meta.get("fit_status"),
        "best_fit_gap":      meta.get("fit_gap"),
        "overfit_threshold":  10.0,
        "underfit_threshold": 70.0,
        "explanation": {
            "GOOD FIT":  "Train-Test F1 gap ≤ 10% and Test F1 ≥ 70%",
            "OVERFIT":   "Train F1 >> Test F1 (gap > 10%): model memorised training data",
            "UNDERFIT":  "Both Train and Test F1 < 70%: model too simple",
        },
        "models": report
    })


@app.route("/api/zones")
def zones():
    return jsonify({"zones": ZONES})


@app.route("/api/alerts")
def alerts():
    import random
    active = []
    rnd = random.Random(int(time.time() // 300))  # Changes every 5 min
    for z in ZONES:
        # Base weather on zone risk
        rain = rnd.uniform(50, 150) if z["base_risk"] == "flood" else rnd.uniform(0, 60)
        feat = {
            "temperature": rnd.uniform(22, 40),
            "rainfall":    round(rain, 1),
            "wind_speed":  rnd.uniform(10, 80),
            "humidity":    rnd.uniform(55, 95),
            "pressure":    rnd.uniform(960, 1015),
            "lat":         z["lat"], "lon": z["lng"],
        }
        pred, proba = _run_model(feat)
        advisories  = _rule_advisories(feat)

        if pred == 1 or advisories:
            disaster = CLASSES[pred] if pred==1 else (advisories[0]+" (Advisory)")
            active.append({
                "zone_id":      z["id"],
                "zone_name":    z["name"],
                "lat":          z["lat"],
                "lng":          z["lng"],
                "disaster_type":disaster,
                "risk_level":   "HIGH" if pred==1 and max(proba)>0.7 else "MODERATE",
                "confidence":   round(max(proba)*100, 1),
                "weather":      feat,
                "issued_at":    datetime.utcnow().isoformat()
            })
    return jsonify({"total_alerts": len(active), "alerts": active})


@app.route("/api/history")
def history():
    import random
    records = []
    rnd     = random.Random(42)
    base    = datetime.now() - timedelta(days=30)
    for i in range(30):
        d    = base + timedelta(days=i)
        rain = rnd.uniform(0, 80)
        feat = {
            "temperature": round(rnd.uniform(20, 42), 1),
            "rainfall":    round(rain, 1),
            "wind_speed":  round(rnd.uniform(0, 70), 1),
            "humidity":    round(rnd.uniform(35, 95), 1),
            "pressure":    round(rnd.uniform(958, 1025), 1),
            "month":       d.month,
            "lat": 28.6, "lon": 77.2,
        }
        pred, proba = _run_model(feat)
        records.append({
            "date":          d.strftime("%Y-%m-%d"),
            "weather":       feat,
            "disaster_type": CLASSES[pred],
            "risk_level":    "HIGH" if pred==1 and max(proba)>0.7 else
                             ("MODERATE" if pred==1 else "LOW"),
            "confidence":    round(max(proba)*100, 1)
        })
    return jsonify({"history": records})


@app.route("/api/model/info")
def model_info():
    return jsonify({
        "best_model":     meta.get("best_model"),
        "accuracy":       meta.get("accuracy"),
        "train_accuracy": meta.get("train_accuracy"),
        "precision":      meta.get("precision"),
        "recall":         meta.get("recall"),
        "f1":             meta.get("f1"),
        "train_f1":       meta.get("train_f1"),
        "auc_roc":        meta.get("auc_roc"),
        "cv_f1_mean":     meta.get("cv_f1_mean"),
        "cv_f1_std":      meta.get("cv_f1_std"),
        "fit_status":     meta.get("fit_status"),
        "fit_gap":        meta.get("fit_gap"),
        "features":       FEATURES,
        "classes":        CLASSES,
        "n_samples":      meta.get("n_samples"),
        "smote_applied":  meta.get("smote_applied"),
        "xgboost_used":   meta.get("xgboost_used"),
        "lightgbm_used":  meta.get("lightgbm_used"),
        "cv_folds":       meta.get("cv_folds"),
        "dataset":        meta.get("dataset_source"),
        "trained_at":     meta.get("trained_at"),
    })


if __name__ == "__main__":
    print("BHU SURAKSHA API v5.0 starting on http://0.0.0.0:5000")
    print(f"OpenWeatherMap key: {OWM_API_KEY[:8]}...{OWM_API_KEY[-4:]}")
    app.run(host="0.0.0.0", port=5000, debug=True)
