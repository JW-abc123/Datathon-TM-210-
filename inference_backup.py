import pandas as pd
import joblib
import argparse

def main(args):
    # 1. Load test data
    test_data = pd.read_csv(args.input)
    print("Test data shape:", test_data.shape)

    # 2. Load trained model
    model = joblib.load(args.model)
    print("Model loaded.")

    # 3. Predict
    predictions = model.predict(test_data)

    # 4. Save to submission file
    submission = pd.DataFrame({
        "id": test_data.index,      # or test_data["id"] if you have an id column
        "prediction": predictions
    })
    submission.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gb_model.pkl", help="Path to trained model")
    parser.add_argument("--input", type=str, default="test.csv", help="Path to input test data")
    parser.add_argument("--output", type=str, default="submission.csv", help="Path to save predictions")
    args = parser.parse_args()
    main(args)

python inference.py --model gb_model.pkl --input test.csv --output submission.csv
