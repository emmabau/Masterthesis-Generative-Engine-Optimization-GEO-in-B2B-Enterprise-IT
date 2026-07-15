"""Replaces duplicate or invalid prompts in the dataset CSV.

This script scans the master dataset for numeric anomalies or duplicate user
prompts and substitutes them with fresh, predefined prompts. It outputs a
corrected CSV and logs the modified row IDs for subsequent scraping.
"""

import csv

new_prompts = [
    "What are the most critical steps for a BDM to take when acting upon an Exascale implementation plan?",
    "As an SME, how do I go about actually deploying an Exascale architecture in our data center?",
    "Are we getting the expected value out of our Exascale systems as end users?",
    "I'm an SME looking for advanced virtualization solutions to dramatically increase our workload density.",
    "Our servers are overheating and as an end user, I really want a direct liquid cooling system installed.",
    "Is direct liquid cooling still the best option for our data center, or are there newer thermal management techniques?",
    "We need to procure reliable B2B-OEM solutions that can seamlessly integrate into our enterprise hardware.",
    "As an end user, I desperately need a business continuity solution that keeps my applications online 24/7.",
    "How can I advise my clients to maximize the ROI of their B2B-OEM solutions after the initial deployment?",
    "How do we as BDMs ensure our business continuity plan remains effective year after year?",
    "What are the best practices for an ITDM to optimize an existing hyperconverged infrastructure?",
    "I need an automated data backup solution so I never lose my important files again.",
    "How do I trigger our disaster recovery procedures if my primary system goes down?",
    "What are the most cutting-edge enterprise data storage solutions available for my clients?",
    "As an end user, how do I configure the new HCI environment we just purchased?",
    "I need a comprehensive overview of IoT implementation strategies in industrial sectors to present to clients.",
    "What are the best practices for maintaining direct liquid cooling systems in client data centers?",
    "What are the exact technical steps to integrate B2B-OEM hardware components into our custom builds?",
    "I want to partner with a top-tier B2B-OEM provider to scale our product offerings globally.",
    "We need virtualization software that significantly reduces our hardware expenditures without sacrificing performance.",
    "What is the step-by-step process for migrating legacy servers to a new HCI platform?",
    "I need to compare SLA guarantees of leading data storage vendors before we sign the final contract.",
    "Can someone provide a technical architecture breakdown of modern HCI platforms?",
    "Are our current compliance service providers delivering enough value against the latest regulatory updates?",
    "Where can I find detailed performance benchmarks for the latest Exascale computing processors?",
    "What are the fundamentals of building resilient IT architectures for ensuring business continuity?",
    "We need to design a highly available architecture to guarantee absolute zero downtime.",
    "I need an overview of IT asset management frameworks to recommend to my enterprise clients.",
    "I want to help my clients implement secure and scalable IoT ecosystems.",
    "How do I provision resources using our new composable infrastructure self-service portal?",
    "Should my clients upgrade their existing direct liquid cooling setups to two-phase systems?",
    "Is the ROI on our direct liquid cooling investment justifying the high maintenance costs?",
    "I need to evaluate the long-term cost benefits of various enterprise IoT platforms.",
    "What is the tangible business value of adopting a hyperconverged infrastructure model?",
    "I need an implementation guide for installing direct liquid cooling in high-density server racks.",
    "Should we completely replace our on-premise data storage with a cloud-based solution?",
    "Is our current security framework adequate to protect against the latest ransomware threats?",
    "Do we need to upgrade our Exascale cluster to support the new generation of AI workloads?",
    "How does two-phase direct liquid cooling compare to single-phase systems in terms of efficiency?",
]

# Note: We provided 39 just to be safe. We only need 24.

input_file = "masterarbeit_dataset_systematisch.csv"
output_file = "masterarbeit_dataset_systematisch_fixed.csv"

seen_prompts = set()
replaced_count = 0
rows_to_scrape = []  # list of IDs to scrape

with (
    open(input_file, "r", encoding="utf-8-sig") as infile,
    open(output_file, "w", encoding="utf-8-sig", newline="") as outfile,
):
    reader = csv.DictReader(infile, delimiter=";")
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()

    for i, row in enumerate(reader):
        prompt = row["User Prompt"]
        # Check if it's a digit or a duplicate
        if prompt.isdigit() or prompt in seen_prompts:
            new_p = new_prompts[replaced_count]
            row["User Prompt"] = new_p
            replaced_count += 1
            rows_to_scrape.append((i + 1, new_p))  # ID is 1-indexed, i starts at 0
            seen_prompts.add(new_p)
        else:
            seen_prompts.add(prompt)

        writer.writerow(row)

print(f"Replaced {replaced_count} faulty prompts.")

# Write the IDs to scrape to a text file for the next script
with open("scratch/rows_to_scrape.txt", "w", encoding="utf-8") as f:
    for idx, p in rows_to_scrape:
        f.write(f"{idx}|||{p}\n")
