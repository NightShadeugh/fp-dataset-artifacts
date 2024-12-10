import pandas as pd
import json
import argparse

def filter_incorrect_predictions(file_path):
    """
    Reads a JSONL file and creates a DataFrame for incorrect predictions.

    Args:
        file_path (str): Path to the JSONL file.
    """
    data = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            # if predicted_label is incorrect
            if record["label"] != record["predicted_label"]:
                data.append(record)
    
    df = pd.DataFrame(data)
    print(f"Filtered DataFrame contains {len(df)} incorrect predictions.")
    return df

def main():
    parser = argparse.ArgumentParser(description="Filter incorrect predictions from a JSONL file.")
    parser.add_argument("--file", required=True, help="Path to the JSONL file.")
    args = parser.parse_args()

    # filtering incorrect predictions
    file_path = args.file
    df = filter_incorrect_predictions(file_path)

    print(df)

if __name__ == "__main__":
    main()