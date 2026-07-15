"""Counts the number of unique prompts within the scraped dataset.

This script parses metadata files across multiple prompt directories to extract
the prompt text and compute the total number of unique prompts present.
"""

import json
import os

g = set()
for i in range(1, 601):
    p = os.path.join("scraped_data_full", f"prompt_{i:03d}", "metadata.json")
    try:
        d = json.load(open(p, encoding="utf-8"))
        pt = d.get("prompt")
        g.add(pt)
    except Exception:
        pass

print(len(g))
