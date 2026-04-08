# Deutsche Bahn Train Delay Prediction

A pre-trip delay risk estimator for Deutsche Bahn trains. Given a planned journey — station, destination, train type, day, and time — the system predicts the probability of significant delay before real-time data is available.
> **This is not a real-time tracker.** DB's own app already does that better. This is a planning tool — it answers *"how risky is this journey historically?"* at booking time.


---

## What It Predicts

| Output | Description |
|---|---|
| Any delay (>0 min) | Historical delay rate for this route type |
| Miss connection (>=6 min) | DB official delay threshold — main model output |
| Plans disrupted (>=15 min) | Serious disruption probability |
| Typical / worst-case range | Median and 90th percentile delay in minutes |

---

## Dataset

- 2,054,632 departure records across 16 German states and 1,994 stations
- Date range: July 8–15, 2024
- Overall significant delay rate: 5.4% (DB threshold: >=6 minutes)

---

## Model Results

| Model | ROC-AUC | Gap Score |
|---|---|---|
| LightGBM | 0.7655 | 0.0355 |
| XGBoost | 0.7581 | 0.0386 |
| CatBoost | 0.7399 | 0.0645 |
| Random Forest | 0.7403 | 0.0587 |

Best model selected by AUC score (rewards stable generalisation over high training scores).
Why ROC-AUC and not accuracy? With only 5.4% delayed trains, a model predicting "on time" for everything achieves 94.6% accuracy while being completely useless. ROC-AUC measures ranking ability — whether the model correctly identifies which journeys are riskier than others.

---

## Tech Stack

Python, scikit-learn, XGBoost, LightGBM, CatBoost, Flask, Docker, AWS EC2, Amazon ECR, GitHub Actions


---

## Run Locally

```bash
git clone https://github.com/dixitdevarshi/Deutsche-Bahn-Delay-Prediction
cd Deutsche-Bahn-Delay-Prediction
pip install -r requirements.txt
python app.py
```

---

## Deployment on AWS EC2 with Docker

### EC2 Setup

```bash
sudo apt-get update -y
sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### Configure EC2 as Self-Hosted GitHub Runner

Go to repository Settings → Actions → Runners → New self-hosted runner and follow the commands to register the EC2 instance.

### GitHub Secrets Required

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
AWS_ECR_LOGIN_URI
ECR_REPOSITORY_NAME
```

Every push to `main` triggers: lint → build Docker image → push to ECR → deploy on EC2.

---

Known Limitations
1. Date range: Only 8 days of data (July 8–15, 2024). More dates would improve model stability.
2. Delay duration (regression): R² ≈ 0.02 across all regressors — duration is driven by real-time network state not available at booking time. Replaced with historical quantile ranges.
3. Station matching: 1,994 stations in lookup. Unknown stations fall back to Germany centre coordinates.
4. `num_stops` approximation: Counts stops on the full line, not just the user's specific segment.
5. Temporal scope: Model reflects July 2024 patterns. Seasonal variation not captured.

---

## Author

Devarshi Dixit
M.Sc. Intelligent Interactive Systems, Universität Bielefeld

