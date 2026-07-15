"""Analyzes the quality of scraped web data by checking domains and file lengths.

This script iterates through a directory of scraped data prompts, identifies
sources from excluded domains (e.g., social media or Q&A sites), and compiles a
quality report detailing the occurrences of excluded domains and the shortest
files that may represent low-quality data.
"""

import json
import os
from collections import Counter
from urllib.parse import urlparse


OUTPUT_DIR = "scraped_data_full"
exclude_list = [
    "google",
    "facebook",
    "twitter",
    "instagram",
    "tiktok",
    "quora",
    "linkedin",
]

bad_domains_found = []
all_sources = []

for i in range(1, 601):
    prompt_dir = os.path.join(OUTPUT_DIR, f"prompt_{i:03d}")
    metadata_file = os.path.join(prompt_dir, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for source in data.get("sources", []):
                    url = source.get("url", "").lower()
                    char_count = source.get("character_count", 0)

                    try:
                        domain = urlparse(url).netloc
                        # check if domain contains any excluded keyword
                        if any(ex in domain for ex in exclude_list):
                            bad_domains_found.append((domain, url, char_count, i))
                    except Exception:
                        pass

                    all_sources.append(
                        {
                            "prompt_id": i,
                            "url": url,
                            "char_count": char_count,
                            "filename": source.get("filename"),
                        }
                    )
            except Exception:
                pass

with open(
    "C:\\Users\\finnb\\Documents\\emma\\scratch\\quality_report_utf8.txt",
    "w",
    encoding="utf-8",
) as out_f:
    out_f.write(f"Total sources analyzed: {len(all_sources)}\n")
    out_f.write("\n--- Social Media / QA Domains found ---\n")
    domain_counts = Counter([d[0] for d in bad_domains_found])
    for dom, count in domain_counts.items():
        out_f.write(f"{dom}: {count} times\n")

    out_f.write("\n--- Top 20 Shortest Files (Potential Garbage) ---\n")
    all_sources.sort(key=lambda x: x["char_count"])
    for src in all_sources[:50]:
        out_f.write(
            f"Prompt {src['prompt_id']:03d} | {src['char_count']} chars | {src['url']}\n"
        )
