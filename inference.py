import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import sys
import matplotlib.pyplot as plt

classes = ['Normal', 'Suspect', 'Pathological']

def load_model(model_path="xgb_opt.json"):
    """Load a trained XGBoost model from JSON."""
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def explain(model, X, preds, class_idx=-1, sample_idx = 0):
    explainer = shap.Explainer(model)
    shap_exp = explainer(X)

    sample = X.iloc[sample_idx]
    values = shap_exp.values[sample_idx] # shape (features, classes)
    base_values = shap_exp.base_values[sample_idx] 
    
    title = f"SHAP Waterfall Plot for Sample {sample_idx}, Class "
    if class_idx == -1:
        class_idx = preds[sample_idx]
        title = title + classes[class_idx].upper() + " (Predicted)"
    else:
        title = title + classes[class_idx].upper()

    # Draw plot for class_idx
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall( 
        shap.Explanation(
            values=values[:, class_idx], 
            base_values=base_values[class_idx],
            data=sample,
            feature_names=X.columns, 
        ),
        max_display=15,
        show = False
    )

    plt.title(title)
    plt.show()

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

    # Make sure to drop label columns if they exist
    X = data.drop(columns=["target"], errors="ignore")

    model = load_model(model_path)
    preds = predict(model, X).astype(int)

    class_idx = -1
    sample_idx = 0

    if len(sys.argv) >= 3: class_idx = int(sys.argv[2])  # Optional class index for explanation
    if len(sys.argv) == 4: sample_idx = int(sys.argv[3])  # Optional sample index for explanation
    
    explain(model, X, preds, class_idx, sample_idx)

    # Output predictions
    print("Predictions:")
    print(preds[:10])  # show first 10
    np.savetxt("predictions.txt", preds, fmt="%d")
    print("Saved predictions to predictions.txt")