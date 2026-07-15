"""Verifies the completeness of the dataset's topical combinations.

This script cross-references the existing rows in the master dataset CSV against
all theoretically possible combinations of focus topics, journey phases, and
personas to identify missing or duplicated entries.
"""

import csv
import itertools

# Definitions from dataset synthese copy.py
FOCUS_TOPICS = [
    "Business Continuity",
    "B2B-OEM Solutions",
    "Composable Infrastructure",
    "Data Backup",
    "Data Storage",
    "Direct Liquid Cooling",
    "Exascale",
    "Finance & Asset Management Services",
    "High-Performance Computing (HPC)",
    "Hyperconverged Infrastructure (HCI)",
    "Hybrid Workplace",
    "Internet of Things (IoT)",
    "Security/Risk/Compliance Services",
    "Server Management",
    "Virtualization Solutions",
]

JOURNEY_PHASES = [
    "Awareness",
    "Interest",
    "Desire",
    "Action",
    "Post-Purchase Use",
    "Post-Decision Info Search",
    "Re-evaluation",
    "No-Funnel",
]

PERSONAS = [
    "Business Decision Maker (BDM)",
    "Subject Matter Expert (SME)",
    "IT Decision Maker (ITDM)",
    "End User",
    "External Consultant",
]

# Create all expected combinations
expected_combinations = set(itertools.product(FOCUS_TOPICS, JOURNEY_PHASES, PERSONAS))

CSV_FILENAME = r"C:\Users\finnb\Documents\emma\masterarbeit_dataset_systematisch.csv"

actual_combinations = set()
duplicates = []

try:
    with open(CSV_FILENAME, mode="r", encoding="utf-8-sig") as f:
        # Using a custom reader because the file uses semicolon and potentially quotes
        reader = csv.reader(f, delimiter=";", quotechar='"')
        header = next(reader)  # skip header

        for row in reader:
            if len(row) < 4:
                continue
            # Row format: User Prompt; Focus Topic; Journey Phase; Persona
            topic = row[1].strip()
            phase = row[2].strip()
            persona = row[3].strip()

            combo = (topic, phase, persona)
            if combo in actual_combinations:
                duplicates.append(combo)
            actual_combinations.add(combo)

    missing_combinations = expected_combinations - actual_combinations

    print(f"Total expected combinations: {len(expected_combinations)}")
    print(f"Total actual combinations found: {len(actual_combinations)}")
    print(f"Missing combinations: {len(missing_combinations)}")

    if missing_combinations:
        print("\nFirst 10 missing combinations:")
        for combo in list(missing_combinations)[:10]:
            print(f"  - Topic: {combo[0]}, Phase: {combo[1]}, Persona: {combo[2]}")

    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate combinations in the CSV.")

except Exception as e:
    print(f"Error reading CSV: {e}")
