"""Identifies duplicate user prompts in the systematic dataset.

This script reads a CSV file containing dataset prompts, detects duplicate
entries based on the user prompt string, and outputs detailed information
regarding the duplicates found.
"""

import csv

d = {}
with open("masterarbeit_dataset_systematisch.csv", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f, delimiter=";")
    for i, row in enumerate(reader):
        d.setdefault(row["User Prompt"], []).append((i + 2, row))

dups = {k: v for k, v in d.items() if len(v) > 1}
total_extra = sum(len(v) - 1 for v in dups.values())

print(f"Found {total_extra} extra rows from {len(dups)} duplicated strings.")
for k, v in dups.items():
    print(f"\nOriginal: {k}")
    for idx, row in v:
        print(
            f"  Row {idx}: {row['Focus Topic']} | {row['Journey Phase']} | {row['Persona']}"
        )
