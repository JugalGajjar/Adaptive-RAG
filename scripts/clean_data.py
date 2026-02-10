"""
Removing duplicates and null contexts from the JSONL dataset.
"""

import json

def clean_data(input_file, output_file):
    seen = set()
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            if not data.get("context"):
                continue

            q = data.get("question", "")

            if q not in seen:
                seen.add(q)
                outfile.write(json.dumps(data) + "\n")

    print(f"Cleaned data written to {output_file}")

if __name__ == "__main__":
    files = ["data/train.jsonl", "data/val.jsonl", "data/test.jsonl"]
    
    for f in files:
        output_file = f.replace(".jsonl", "_cleaned.jsonl")
        clean_data(f, output_file)