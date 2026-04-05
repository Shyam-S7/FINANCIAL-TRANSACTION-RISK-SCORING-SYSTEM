# Financial Transaction Risk Scoring System

An industry-standard, End-to-End Machine Learning architecture designed to predict financial transaction fraud using robust, decoupled ML components and serving mechanisms.

## 🏗 Modular Architecture Flow

The system is fully modular, configuration-driven, and decoupled into discrete micro-operational stages:

1. **Data Ingestion**: Loads data dynamically from `config.yaml` pathways, asserting the input and generating structured Train and Test splits.
2. **Data Validation**: Confirms that data schemas conform strictly to expected properties dictating `schema.yaml` preventing downstream pipeline execution failures structurally. 
3. **Data Transformation & Feature Engineering**: Creates domain-specific features (e.g., `errorBalanceOrg` deriving the transaction discrepancies) and channels everything through isolated pre-processing pipelines safely preserving categorical encoders and imputers locally. 
4. **Model Trainer**: Cross-validates via iterative GridSearch sweeps across algorithms like Logistic Regression, Random Forests, and XGBoost models. Determines the maximum performance logic and generates an offline artifact safely.
5. **Model Evaluation**: Re-tests hold-out samples measuring industry standards: Precision, Recall, F1-Scores, and aggregate ROC-AUC scores.
6. **Prediction & API Services**: A containerized standard `FastAPI` instance structured independently. The architecture decouples prediction pipelines exclusively targeting inference loads on incoming JSON objects preventing any bloated memory instances.

## 💻 Technical Infrastructure
* **Language**: Python 3.9
* **Machine Learning**: Scikit-Learn, XGBoost
* **API Framework**: FastAPI, Pydantic 
* **Containerization**: Docker
* **Configurations**: PyYAML / Dataclasses

---

## 🚀 Setup & Execution

You will need the raw dataset saved inside the folder path instructed within your `config.yaml` (`data/transactions.csv`). The pipeline will generate the appropriate missing files/schema during testing if it isn't completely present. 

### Local System Execution
```bash
# 1. Initialize environment using uv
uv venv

# Windows activate:
.venv\Scripts\activate
# Linux/Mac activate:
# source .venv/bin/activate

# 2. Extract dependencies natively with blazing fast uv pip
uv pip install -r requirements.txt

# 3. Formulate models
python -m src.pipeline.training_pipeline

# 4. Initiate API Service 
python api/app.py 
```

### Docker Execution
```shell
docker build -t risk-scoring-system .
docker run -p 8000:8000 risk-scoring-system
```

---

## 📡 API Inference Usage 

Provide a POST request structured exactly like so directly accessing `http://localhost:8000/predict`:

**Input Payload**
```json
{
  "amount": 5000.50,
  "oldbalanceOrg": 10000.00,
  "newbalanceOrig": 4999.50,
  "oldbalanceDest": 100.0,
  "newbalanceDest": 5100.50,
  "type": "TRANSFER"
}
```

**Output Expected**
```json
{
  "fraud_probability": 0.9840,
  "risk_label": "High"
}
```
