<div align="center">

# 🔍 Real-Time Fake Job Detector

**A production-grade streaming data pipeline that detects fraudulent job postings in real time.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Apache Kafka](https://img.shields.io/badge/Apache_Kafka-3.8-231F20?style=for-the-badge&logo=apachekafka&logoColor=white)](https://kafka.apache.org)
[![Apache Spark](https://img.shields.io/badge/Apache_Spark-3.5.3-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [ML Model](#-ml-model-performance) • [Testing](#-testing) • [API Docs](#-api-reference) • [Contributing](#-contributing)

</div>

---

## 📌 Overview

Job boards are flooded with fraudulent postings designed to steal personal data or money. This system automatically flags fake jobs **the moment they are posted** — no manual review required.

Every job posting flows through a Kafka queue → Spark ML pipeline → PostgreSQL → FastAPI → live Next.js dashboard. Spark scores postings in **5-second micro-batches**, and the model is tuned for **F1 on heavily imbalanced data** (4.8% fraud): **97.8% accuracy, 75.6% F1, 82% precision**.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| ⚡ **Real-time streaming** | Kafka → Spark micro-batches every 5 seconds |
| 🤖 **ML inference at scale** | TF-IDF + class-weighted Random Forest UDF runs across Spark workers |
| ⚖️ **Imbalance-aware** | Class weighting + validation-tuned decision threshold, F1 as primary metric |
| 📈 **Anomaly detection** | EMA-based spike detection flags unusual fraud surges |
| 🔗 **Fraud clustering** | Cosine similarity groups related fake jobs (≥ 0.8 threshold) |
| ✏️ **Human corrections** | Manual override system with full correction audit trail |
| 📡 **Live dashboard** | WebSocket-powered Next.js frontend with real-time charts |
| 🪦 **Dead letter queue** | Failed events captured for manual investigation |
| 🐳 **One-command deploy** | Full Docker Compose stack — 7 services, one command |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA FLOW                                    │
│                                                                   │
│  CSV Dataset                                                      │
│      │                                                            │
│      ▼                                                            │
│  ┌──────────────┐     ┌──────────────────┐                       │
│  │ Kafka        │────►│  Kafka Topic     │                       │
│  │ Producer     │     │  (jobs)          │                       │
│  └──────────────┘     └────────┬─────────┘                       │
│                                │                                  │
│                                ▼                                  │
│                  ┌─────────────────────────┐                     │
│                  │  Spark Structured       │                     │
│                  │  Streaming              │                     │
│                  │  ├─ TF-IDF + RF model   │                     │
│                  │  ├─ Anomaly detection   │                     │
│                  │  └─ Fraud clustering    │                     │
│                  └────────────┬────────────┘                     │
│                               │                                   │
│                               ▼                                   │
│                  ┌─────────────────────────┐                     │
│                  │     PostgreSQL           │                     │
│                  │  ├─ job_predictions      │                     │
│                  │  ├─ job_clusters         │                     │
│                  │  ├─ job_corrections      │                     │
│                  │  └─ pipeline_metrics     │                     │
│                  └────────────┬────────────┘                     │
│                               │                                   │
│                               ▼                                   │
│                  ┌─────────────────────────┐                     │
│                  │  FastAPI                 │                     │
│                  │  REST + WebSocket        │                     │
│                  └────────────┬────────────┘                     │
│                               │                                   │
│                               ▼                                   │
│                  ┌─────────────────────────┐                     │
│                  │  Next.js Dashboard       │                     │
│                  │  Live charts & alerts    │                     │
│                  └─────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### Services

| Service | Image | Port | Role |
|---------|-------|------|------|
| `fraud-postgres` | postgres:16-alpine | 5432 | Predictions database |
| `fraud-kafka` | apache/kafka:3.8 | 9092 | KRaft message broker |
| `fraud-spark-master` | custom | 7077 / 8080 | Spark cluster coordinator |
| `fraud-spark-worker` | custom | 8081 | ML inference executor |
| `fraud-spark-stream` | custom | 4040 | Streaming driver |
| `fraud-api` | custom | 8000 | FastAPI REST + WebSocket |
| `fraud-dashboard` | custom | 3001 | Next.js frontend |

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop 24+ 
- 6 GB RAM available
- 10 GB free disk space

```bash
# 1. Clone the repo
git clone https://github.com/sriramm84/Realtime-Fake-Job-Predictor.git
cd Realtime-Fake-Job-Predictor

# 2. Set up environment
cp .env.example .env
# ⚠️  Edit .env and set a strong POSTGRES_PASSWORD

# 3. Launch everything
docker compose up --build
```

| Service | URL |
|---------|-----|
| 📊 Live Dashboard | http://localhost:3001 |
| 📖 API Docs (Swagger) | http://localhost:8000/docs |
| ⚡ Spark Master UI | http://localhost:8080 |
| 🔧 Spark Worker UI | http://localhost:8081 |

```bash
# Stop everything
docker compose down

# Also start the Kafka CSV producer (streams dataset into pipeline)
docker compose --profile producer up
```

### Windows Local Dev (no Docker)

```bash
cd fake_job_detector
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python ../start_system.py
```

---

## 🤖 ML Model Performance

Trained on the [Kaggle Fake Job Postings dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction): **17,880 postings, 866 fraudulent (4.8%)**. A model that always says "real" would already score 95.2% accuracy, so **accuracy is not the headline metric here. F1 is.**

<div align="center">

| Metric (held-out test set, 3,576 rows) | Score |
|---|---|
| ⚖️ **F1-score (primary)** | **75.6%** |
| ✅ Accuracy | **97.8%** |
| 🎯 Precision | **82.3%** |
| 🔍 Recall | **70.0%** |

| | Predicted real | Predicted fake |
|---|---|---|
| **Actually real** | 3,377 | 26 |
| **Actually fake** | 52 | 121 |

</div>

Reproduce with `python -m ml.evaluate` (from `fake_job_detector/`); results are written to [`docs/metrics.json`](docs/metrics.json).

### Handling class imbalance

| Technique | Implementation |
|---|---|
| **Class weighting** | `RandomForestClassifier(class_weight="balanced_subsample")`, `LogisticRegression(class_weight="balanced")` |
| **Decision-threshold tuning** | Threshold swept 0.05 to 0.95 on a validation slice carved from the *training* data (the test set is never used for tuning). Best-F1 threshold = **0.28**, saved to `ml/saved_model/threshold.json` and applied identically in offline evaluation, the FastAPI service and the Spark UDF |
| **Stratified splits** | Train/test and train/validation splits preserve the 4.8% fraud rate |
| **F1 for model selection** | Logistic Regression and Random Forest are compared on F1; the winner is persisted |

**Effect of the tuning.** The earlier baseline (unweighted forest, default 0.5 cutoff) had 100% precision but only 60.2% recall (F1 75.2%). Class weighting plus threshold tuning moves recall to 70.0% at 82% precision, which is a better trade-off for a fraud-screening tool, where missed scams cost more than a few extra reviews.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest, tuned threshold** *(selected)* | 97.8% | 82.3% | 70.0% | **75.6%** |
| Logistic Regression, balanced | 95.0% | 48.9% | 93.1% | 64.1% |

### How the model works

1. **Preprocessing:** title + company profile + description + requirements are combined, lower-cased and cleaned
2. **TF-IDF:** 5,000 text features
3. **Engineered features:** `has_company_profile`, `has_salary_range`, `text_length`
4. **Random Forest:** 100 class-weighted trees; fraud when `P(fraud) >= 0.28`

```bash
cd fake_job_detector
python ml/train_model.py      # retrain + retune threshold
python -m ml.evaluate         # reproduce the metrics above
```

---

## ⚡ Performance

| What | Number | How it was obtained |
|---|---|---|
| Model inference, single process, one record at a time | **~32 postings/sec** | Measured by `ml.evaluate` on a dev laptop (CPU only). The same code path runs inside each Spark UDF call, so a single worker core clears a 10 postings/sec stream with headroom |
| Streaming cadence | 5 s micro-batches, up to 700 records per batch (configurable) | `SPARK_TRIGGER_INTERVAL`, `SPARK_MAX_OFFSETS_PER_TRIGGER` |
| End-to-end latency | Recorded per batch as `avg_latency_ms` in `pipeline_metrics` and on the dashboard | Event timestamp → scored timestamp |

End-to-end throughput and latency depend on hardware, so measure them on your machine with the load test below rather than trusting a number in a README.

---

## 📡 API Reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health (Kafka, Spark, DB, ML status) |
| `GET` | `/dashboard` | Aggregated stats snapshot |
| `GET` | `/jobs/latest` | Last 50 processed predictions |
| `GET` | `/jobs/fake` | All flagged fake jobs |
| `GET` | `/alerts` | Top alerts sorted by risk score |
| `GET` | `/trends` | Hourly fake/real job trend (24h window) |
| `GET` | `/clusters/top` | High-pressure fraud clusters |
| `GET` | `/corrections` | Manual human overrides |
| `POST` | `/jobs/ingest` | Submit a new job posting for analysis |
| `WS` | `/ws` | WebSocket — live prediction stream |

Full interactive docs: **http://localhost:8000/docs**

---

## 📁 Project Structure

```
Realtime-Fake-Job-Predictor/
├── docker-compose.yml              # Orchestrates all 7 services
├── .env.example                    # Environment variable template
├── start_system.py                 # Windows local dev launcher
├── DOCKER_SETUP.md                 # Detailed Docker guide
├── docs/metrics.json               # Latest evaluation results
│
└── fake_job_detector/
    ├── api/                        # FastAPI backend
    │   ├── Dockerfile
    │   └── app.py                  # REST endpoints + WebSocket
    │
    ├── dashboard/                  # Next.js real-time frontend
    │   ├── app/                    # Pages: dashboard, alerts, analytics, trends
    │   ├── components/             # UI components
    │   ├── context/                # Dashboard state (WebSocket + polling)
    │   └── services/               # API client
    │
    ├── kafka/                      # Kafka CSV replay producer
    │   └── producer.py
    │
    ├── ml/                         # Model training pipeline
    │   ├── preprocess.py
    │   ├── train_model.py          # class weighting + threshold tuning
    │   ├── evaluate.py             # reproducible metrics -> docs/metrics.json
    │   └── saved_model/            # fraud_model.pkl, tfidf.pkl, threshold.json
    │
    ├── tests/                      # pytest suite
    │
    ├── spark/                      # Spark Structured Streaming job
    │   ├── spark_stream.py         # Main streaming driver
    │   ├── model_loader.py         # ML inference UDF
    │   └── Dockerfile
    │
    ├── utils/                      # DB writer, config, helpers
    └── data/raw/                   # fake_job_postings.csv (17,880 rows)
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_PASSWORD` | *(required)* | Database password |
| `SPARK_TRIGGER_INTERVAL` | `5 seconds` | Spark micro-batch cadence |
| `SPARK_MAX_OFFSETS_PER_TRIGGER` | `700` | Records per Spark batch |
| `ANOMALY_EMA_ALPHA` | `0.2` | EMA smoothing factor |
| `ANOMALY_SPIKE_MULTIPLIER` | `2.2` | Spike threshold multiplier |
| `DASHBOARD_PORT` | `3001` | Host port for Next.js dashboard |
| `REPLAY_SPEED` | `10.0` | CSV replay speed (10x = 10× faster) |
| `MAX_KAFKA_LAG` | `5000` | Producer throttle threshold |

See [`.env.example`](.env.example) for the full list.

---

## 🧪 Testing

```bash
cd fake_job_detector
pip install pytest -r requirements.txt
python -m pytest tests -q
```

The suite was written test-first (RED → GREEN) and covers: class-weight configuration, the F1-maximising threshold sweep and its edge cases (no positives, imbalanced scores), threshold persistence, threshold loading with missing or corrupt files, and threshold-based classification in the inference path. It runs in CI on every push.

### Load testing the live stack

```bash
python fake_job_detector/load_test.py   --total-events 10000   --concurrency 40   --api-base-url http://127.0.0.1:8000
```

Reports success/error counts, throughput (events/sec) and average/max latency.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**If you found this useful, please consider giving it a ⭐**

</div>
