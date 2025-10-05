import xgboost as xgb
import pandas as pd
import numpy as np
import sys

def load_model(model_path="xgb_opt.json"):
    """Load a trained XGBoost model from JSON."""
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def predict(model, X):
    """Make predictions on a Pandas DataFrame or NumPy array."""
    dmatrix = xgb.DMatrix(X)
    preds = model.predict(dmatrix)
    return preds

if __name__ == "__main__":
    # Example usage:
    # python inference.py path/to/data.csv

    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_data.csv>")
        sys.exit(1)

    data_path = sys.argv[1]
    model_path = "xgb_opt.json"

    # Load your CTG-like data
    data = pd.read_csv(data_path)

    # ⚠️ Make sure to drop label columns if they exist
    X = data.drop(columns=["target"], errors="ignore")

    model = load_model(model_path)
    preds = predict(model, X)

    # Output predictions
    print("Predictions:")
    print(preds[:10])  # show first 10
    np.savetxt("predictions.txt", preds, fmt="%d")
    print("Saved predictions to predictions.txt")