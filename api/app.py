import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# Add parent to path for package resolution when running via Uvicorn explicitly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.prediction_pipeline import PredictPipeline, CustomData
from src.utils.logger import logger

app = FastAPI(title="Financial Transaction Risk Scoring API", description="Production API to predict online fraud.")

class TransactionInput(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: str

@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    try:
        # Load transaction data into the OOP custom data object
        data = CustomData(
            amount_val=transaction.amount,
            oldbalanceOrg_val=transaction.oldbalanceOrg,
            newbalanceOrig_val=transaction.newbalanceOrig,
            oldbalanceDest_val=transaction.oldbalanceDest,
            newbalanceDest_val=transaction.newbalanceDest,
            type_val=transaction.type
        )

        df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()

        # Run inference
        preds, probs = predict_pipeline.predict(df)
        
        fraud_prob = float(probs[0])
        label = "High" if preds[0] == 1 else "Low"

        return {
            "fraud_probability": round(fraud_prob, 4),
            "risk_label": label
        }
    except Exception as e:
        logger.error(f"API Error Occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Prediction Error occurred. See logs.")

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
