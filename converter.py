import csv
import json

csv_file_path = "original_300.csv"  # name for the csv
jsonl_file_path = "original_300_set.jsonl"  # name for the jsonl


with open(csv_file_path, mode="r", encoding="utf-8") as csv_file, open(jsonl_file_path, mode="w", encoding="utf-8") as jsonl_file:
    reader = csv.DictReader(csv_file)
    
    for row in reader:
        # CSV columns premise, hypothesis, and label
        jsonl_entry = {
            "premise": row["premise"],
            "hypothesis": row["hypothesis"],
            "label": int(row["label"])
        }

        jsonl_file.write(json.dumps(jsonl_entry) + "\n")

print(f"Conversion DONE! JSONL file saved as {jsonl_file_path}.")
