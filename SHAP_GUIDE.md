# 🔍 SHAP Model Interpretability Guide

## What is SHAP?

**SHAP (SHapley Additive exPlanations)** explains individual predictions by computing how much each feature contributed to the final prediction. It's based on game theory and provides fair, consistent explanations.

---

## ✅ SHAP is Now Integrated!

Your Fire App now includes SHAP for model interpretability:

```
✅ Model loaded successfully (timestamp: 20251224_005344)
✅ SHAP explainer ready for interpretability
✅ ML Risk Model loaded successfully!
```

---

## 🚀 How to Use SHAP

### Option 1: API with SHAP Explanations

Add `explain=true` parameter to get detailed feature contributions:

```bash
curl "http://localhost:5001/api/ml_predict?veg_height=4.5&clearance=2.8&temperature=33&humidity=30&explain=true"
```

### Response with SHAP Explanation:

```json
{
  "success": true,
  "prediction": {
    "risk_level": "High",
    "risk_score": 0.593,
    "risk_percentage": 59.3,
    "confidence": 0.7458,
    "probabilities": {
      "Low": 0.00,
      "Moderate": 0.2542,
      "High": 0.7458,
      "Critical": 0.00
    },
    "explanation": {
      "base_value": 0.285,
      "prediction": 0.593,
      "top_contributors": [
        "veg_to_clearance_ratio",
        "clearance_m",
        "veg_height_m",
        "fire_danger_index",
        "temperature_c"
      ],
      "contributions": {
        "veg_to_clearance_ratio": {
          "value": 1.607,
          "shap_value": 0.158,
          "contribution_pct": 51.3
        },
        "clearance_m": {
          "value": 2.8,
          "shap_value": 0.089,
          "contribution_pct": 28.9
        },
        "veg_height_m": {
          "value": 4.5,
          "shap_value": 0.034,
          "contribution_pct": 11.0
        },
        "fire_danger_index": {
          "value": 0.467,
          "shap_value": 0.015,
          "contribution_pct": 4.9
        },
        "temperature_c": {
          "value": 33.0,
          "shap_value": 0.012,
          "contribution_pct": 3.9
        }
      }
    }
  },
  "has_explanation": true
}
```

---

## 📊 Understanding SHAP Output

### 1. **Base Value**
```json
"base_value": 0.285
```
- The average model prediction across all training data
- Starting point before considering this specific prediction

### 2. **Prediction**
```json
"prediction": 0.593
```
- Final risk score (0-1)
- `base_value + sum(all shap_values) = prediction`

### 3. **Top Contributors**
```json
"top_contributors": [
  "veg_to_clearance_ratio",
  "clearance_m",
  "veg_height_m"
]
```
- Features that had the biggest impact on this prediction
- Ranked by absolute contribution

### 4. **Feature Contributions**

For each feature:

```json
"veg_to_clearance_ratio": {
  "value": 1.607,           // Actual feature value
  "shap_value": 0.158,      // Contribution to prediction
  "contribution_pct": 51.3  // Percentage of total contribution
}
```

**Interpretation:**
- ✅ **Positive SHAP value** = Pushes risk **higher**
- ❌ **Negative SHAP value** = Pushes risk **lower**
- 📊 **Magnitude** = How much it matters

---

## 🎯 Example Use Cases

### Case 1: Understanding a High Risk Prediction

**Input:**
```bash
curl "http://localhost:5001/api/ml_predict?veg_height=5.2&clearance=2.5&temperature=35&humidity=25&explain=true"
```

**SHAP Output:**
```
Top Contributors:
  1. veg_to_clearance_ratio: +0.18  (45% contribution) ⚠️ HIGH RATIO
  2. clearance_m: +0.12            (30% contribution) ⚠️ TOO CLOSE
  3. fire_danger_index: +0.05      (12% contribution) ⚠️ HIGH DANGER
  4. veg_height_m: +0.04           (10% contribution) ⚠️ TALL VEG
  5. humidity_pct: +0.01           (3% contribution)  ⚠️ DRY
```

**Actionable Insight:**
- **Main problem:** Vegetation to clearance ratio (45% of risk)
- **Action:** Increase clearance or reduce vegetation height
- **Impact:** Would reduce risk score by ~0.18 (18 percentage points)

---

### Case 2: Why is This Zone Low Risk?

**Input:**
```bash
curl "http://localhost:5001/api/ml_predict?veg_height=0.8&clearance=7.2&temperature=20&humidity=60&explain=true"
```

**SHAP Output:**
```
Top Contributors:
  1. clearance_m: -0.12           (40% contribution) ✅ GOOD DISTANCE
  2. veg_to_clearance_ratio: -0.09 (30% contribution) ✅ LOW RATIO
  3. humidity_pct: -0.04          (13% contribution) ✅ MOIST
  4. fire_danger_index: -0.03     (10% contribution) ✅ LOW DANGER
  5. temperature_c: -0.02         (7% contribution)  ✅ COOL
```

**Actionable Insight:**
- **Main safety factor:** Good clearance (40% of safety margin)
- **Multiple protective factors** working together
- **Maintain:** Current clearance and vegetation management

---

## 💡 Key Features Explained

### Feature Importance Ranking

1. **veg_to_clearance_ratio** (Derived)
   - Most important feature
   - Captures the critical relationship between vegetation height and distance

2. **clearance_m** (Direct measurement)
   - Distance from vegetation to power line
   - Second most impactful

3. **veg_height_m** (Direct measurement)
   - Absolute vegetation height
   - Third most important

4. **fire_danger_index** (Derived)
   - Combines temperature, humidity, wind, drought
   - Environmental fire risk

5. **temperature_c, humidity_pct, wind_speed_ms**
   - Individual weather factors
   - Smaller but cumulative impact

---

## 🧪 Testing SHAP Locally

### Test in Browser Console

```javascript
// Without explanation
fetch('/api/ml_predict?veg_height=4.5&clearance=2.8&temperature=33&humidity=30')
  .then(r => r.json())
  .then(data => console.log('Prediction:', data.prediction));

// With SHAP explanation
fetch('/api/ml_predict?veg_height=4.5&clearance=2.8&temperature=33&humidity=30&explain=true')
  .then(r => r.json())
  .then(data => {
    console.log('Risk Level:', data.prediction.risk_level);
    console.log('Top Contributors:', data.prediction.explanation.top_contributors);
    console.log('Contributions:', data.prediction.explanation.contributions);
  });
```

---

## 📈 Performance Impact

### Without SHAP (`explain=false` or omitted):
```
Response Time: ~5-10ms
```

### With SHAP (`explain=true`):
```
Response Time: ~50-100ms
```

**Recommendation:** Only use `explain=true` when you need interpretability, not for every request.

---

## 🎨 Visualizing SHAP Values

### Simple Text Visualization

```python
import requests
import json

response = requests.get('http://localhost:5001/api/ml_predict', params={
    'veg_height': 4.5,
    'clearance': 2.8,
    'temperature': 33,
    'humidity': 30,
    'explain': 'true'
})

data = response.json()
explanation = data['prediction']['explanation']

print(f"Base Risk: {explanation['base_value']:.3f}")
print(f"Final Risk: {explanation['prediction']:.3f}")
print(f"\nTop 5 Contributors:")

for feature in explanation['top_contributors'][:5]:
    contrib = explanation['contributions'][feature]
    sign = '+' if contrib['shap_value'] > 0 else ''
    bar = '█' * int(abs(contrib['contribution_pct']) / 2)
    print(f"  {feature:25s} {sign}{contrib['shap_value']:>7.3f} {bar} {contrib['contribution_pct']:.1f}%")
```

**Output:**
```
Base Risk: 0.285
Final Risk: 0.593

Top 5 Contributors:
  veg_to_clearance_ratio    +0.158 █████████████████████████ 51.3%
  clearance_m                +0.089 ██████████████ 28.9%
  veg_height_m               +0.034 █████ 11.0%
  fire_danger_index          +0.015 ██ 4.9%
  temperature_c              +0.012 ██ 3.9%
```

---

## 🔬 Advanced: SHAP in Python

```python
from risk_model import FireRiskModel

# Load model
model = FireRiskModel()
model.load_models()

# Make prediction with explanation
data = {
    'veg_height_m': 4.5,
    'clearance_m': 2.8,
    'temperature_c': 33,
    'humidity_pct': 30,
    'wind_speed_ms': 12,
    'days_since_rain': 15,
    'latitude': 37.5,
    'longitude': -121.8
}

# Get prediction with SHAP values
prediction = model.predict(data, explain=True)

print(f"Risk Level: {prediction['risk_level']}")
print(f"Risk Score: {prediction['risk_score']:.3f}")
print(f"\nTop Features:")
for feature in prediction['explanation']['top_contributors'][:5]:
    contrib = prediction['explanation']['contributions'][feature]
    print(f"  {feature}: {contrib['shap_value']:+.4f} ({contrib['contribution_pct']:.1f}%)")
```

---

## ❓ FAQ

### Q: What if `explanation` is not in the response?

**A:** Either:
1. You didn't include `explain=true` parameter
2. SHAP explainer wasn't initialized (retrain the model)

### Q: Why are some SHAP values negative?

**A:** 
- **Negative** = Feature reduces risk
- **Positive** = Feature increases risk
- **Zero** = No impact

### Q: Can I use SHAP for batch predictions?

**A:** Currently, SHAP is only available for single predictions via `/api/ml_predict`. For performance reasons, batch predictions don't include SHAP by default.

### Q: How accurate are SHAP values?

**A:** SHAP provides **exact** feature attributions based on game theory. They are mathematically proven to be fair and consistent.

---

## 🎉 Summary

✅ **SHAP is integrated** into your Fire App
✅ **Add `explain=true`** to get feature contributions
✅ **Understand WHY** the model makes predictions
✅ **Take action** based on top contributors
✅ **Scientifically sound** explanations (Shapley values)

**Your ML model is now fully interpretable and explainable!** 🚀

---

## 📚 Learn More

- [SHAP Documentation](https://shap.readthedocs.io/)
- [Original Paper](https://arxiv.org/abs/1705.07874)
- [Tutorial](https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)



