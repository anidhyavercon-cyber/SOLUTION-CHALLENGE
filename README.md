# BHU SURAKSHA 🌊⛰🌀
**Disaster Prediction System — Uttarakhand, India**

ML-powered flood, landslide & cyclone prediction with live map and risk-zone registry.

---

## 📁 Project Structure
```
BHU_SURAKSHA/
├── backend/
│   └── app.py               ← Flask REST API (all endpoints)
├── training/
│   ├── train_model.py        ← Train / retrain the ML model
│   └── generate_and_train.py ← Generates synthetic dataset + trains
├── data/
│   ├── weather_data.csv            ← Original dataset (200 rows)
│   └── weather_data_enhanced.csv   ← Generated rich dataset (5000 rows)
├── models/
│   ├── disaster_model.pkl   ← Trained Voting Ensemble
│   ├── scaler.pkl           ← StandardScaler
│   └── model_meta.json      ← Accuracy, features, classes
├── frontend/
│   └── index.html           ← Full dashboard (no build needed)
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (first time)
```bash
cd training
python train_model.py
```
Expected output: **✅ Test Accuracy: 99.83%**

### 3. Start backend
```bash
cd backend
python app.py
```
Backend runs on `http://localhost:5000`

### 4. Open frontend
Open `frontend/index.html` in any browser.

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check + model accuracy |
| `/api/predict` | POST | Single prediction (JSON body) |
| `/api/predict/batch` | POST | Batch prediction (CSV upload or JSON array) |
| `/api/zones` | GET | All risk zones |
| `/api/zones/<id>/risk` | GET | Live risk for a specific zone |
| `/api/alerts` | GET | Active disaster alerts |
| `/api/history` | GET | 30-day prediction history |
| `/api/model/info` | GET | Model details & accuracy |

### Predict (POST /api/predict)
```json
{
  "temperature": 32,
  "rainfall": 90,
  "wind_speed": 45,
  "humidity": 88,
  "pressure": 1000,
  "river_level": 14,
  "soil_moisture": 80
}
```

---

## 🧠 ML Model

- **Algorithm**: Voting Ensemble (GradientBoostingClassifier + RandomForestClassifier)
- **Accuracy**: **99.83%** on test set, **>99%** 5-fold CV
- **Minimum target**: 96% ✅
- **Features**: temperature, rainfall, wind_speed, humidity, pressure, river_level, soil_moisture
- **Classes**: None (0), Flood (1), Landslide (2), Cyclone (3)
- **Training samples**: 3000–5000 synthetic records with realistic physical rules

---

## 🗺 Risk Zones (Uttarakhand)

| ID | Zone | Risk Type | No Construction |
|---|---|---|---|
| Z1 | Haridwar – Ganga Floodplain | Flood | ✅ |
| Z2 | Rishikesh Valley | Flood | ❌ |
| Z3 | Kedarnath Region | Landslide | ✅ |
| Z4 | Uttarkashi | Landslide | ✅ |
| Z5 | Dehradun City Core | None | ❌ |
| Z6 | Chamoli – Nanda Devi | Landslide | ✅ |
| Z7 | Pithoragarh Hills | Landslide | ❌ |
| Z8 | Tehri Dam Reservoir | Flood | ✅ |

---

## 🔧 To Add Real Weather Data
In `backend/app.py`, replace the `sim = {...}` block in `/api/zones/<zone_id>/risk`
with a real call to the IMD or OpenWeatherMap API.

---

## 📞 Emergency Contacts
- NDMA Helpline: **1078**
- Uttarakhand SDRF: **9557444486**
- IMD: **imd.gov.in**
