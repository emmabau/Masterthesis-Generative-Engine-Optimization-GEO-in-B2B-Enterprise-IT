"""Generates a dataset of user queries for enterprise IT scenarios.

This module automates the creation of a systematic dataset containing user queries.
It combines focus topics, customer journey phases, and buyer personas, requesting
a language model to generate corresponding queries.
"""

import itertools
import random
import time

import requests

# --- 1. KONFIGURATION ---
API_KEY = "JQY53B7-MVZ4K40-Q7RJFPJ-DAXMVRB"
BASE_URL = "http://localhost:3001/api/v1"
WORKSPACE_SLUG = "my-workspace"

CSV_FILENAME = "masterarbeit_dataset_systematisch.csv"

# --- 2. DATENSTRUKTUREN ---
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

JOURNEY_DESCRIPTIONS = {
    "Awareness": "Early phase, broad search, problem recognition.",
    "Interest": "Information search, looking at alternatives (middle stage).",
    "Desire": "Transactional intent, comparing specific vendors, deep evaluation.",
    "Action": "Final selection, purchasing logistics, closing.",
    "Post-Purchase Use": "Implementation, troubleshooting, daily operations.",
    "Post-Decision Info Search": "Validating the choice, looking for advanced documentation.",
    "Re-evaluation": "Renewal cycles, expansion, or considering switching.",
    "No-Funnel": "General industry interest, research-driven, not currently in a buying cycle.",
}

PERSONA_DESCRIPTIONS = {
    "Business Decision Maker (BDM)": "Focus on ROI, TCO, business impact, and strategic growth.",
    "Subject Matter Expert (SME)": "Focus on team impact, functional utility, and technical viability.",
    "IT Decision Maker (ITDM)": "Focus on integration, security standards, compliance, and efficiency.",
    "End User": "Focus on usability, workflow, and individual productivity.",
    "External Consultant": "Focus on strategy, requirements mapping, and vendor benchmarking.",
}

VARIATION_TEMPLATES = [
    "Use highly technical, niche IT jargon. Tone: Expert talking to expert.",
    "Keep prompts very short and direct, like a quick support ticket.",
    "Describe complex, multi-layered strategic challenges. Tone: C-Level Executive.",
    "Write from a perspective of frustration – facing real technical roadblocks.",
    "Focus on future-proofing, budgets and compliance regulations (e.g. NIS2, GDPR).",
    "Use a mix of formal inquiries and informal, internal team chat styles.",
]

# --- 3. DATA GENERATION FUNCTION ---


def generate_dataset_batch(batch_num, combinations_chunk, previous_examples):
    """Generate a batch of unique dataset entries via the language model API.

    Args:
        batch_num (int): The index of the current generation batch.
        combinations_chunk (list of tuple): A list of tuples containing (topic, phase, persona).
        previous_examples (list of str): A list of previously generated prompts to prevent duplication.

    Returns:
        str: The raw generated text in CSV format, or None if the request fails.
    """
    selected_style = random.choice(VARIATION_TEMPLATES)

    # Construct an exact list for the language model to process in this batch
    tasks_str = ""
    for idx, combo in enumerate(combinations_chunk):
        topic, phase, persona = combo
        tasks_str += (
            f"{idx + 1}. Topic: {topic} | Phase: {phase} | Persona: {persona}\n"
        )

    avoidance_text = ""
    if previous_examples:
        avoidance_text = "CRITICAL AVOIDANCE RULE: Do NOT use the exact same phrasing or starting words as these examples from the previous batch:\n"
        for ex in previous_examples:
            avoidance_text += f"- {ex}\n"

    prompt_text = f"""
Role: Expert Prompt Engineer for Enterprise IT.
Task: Generate exactly {len(combinations_chunk)} unique dataset entries. 

YOUR EXACT ASSIGNMENTS FOR THIS BATCH:
You must generate exactly one row for each of the following combinations, in this exact order:
{tasks_str}

CONTEXT DEFINITIONS:
- Reference the Journey Phases and Personas accurately based on Enterprise IT standards.

VARIATION RULES:
{selected_style}
{avoidance_text}
- Never start more than two prompts with "What is" or "How to".
- Temperature is set to 0.9.
- Ensure the 'User Prompt' sounds exactly like the assigned Persona in that Journey Phase.
- No repetitions of phrasing from previous entries.
- Never start more than 2 prompts in this batch with the same word.
- Avoid 'What is', 'How to', 'I need'.
- Use different sentence types: imperative ('Analyze this...'), situational ('We are currently facing...'), and conversational ('Someone told me that...')."

OUTPUT FORMAT:
Generate ONLY the raw CSV text. Use a semicolon (;) as the delimiter.
Format: "User Prompt";"Focus Topic";"Journey Phase";"Persona"
Ensure the Topic, Phase, and Persona match your assignments exactly.
Do NOT include a header row. Do NOT wrap it in Markdown code blocks like ```csv.
"""

    url = f"{BASE_URL}/workspace/{WORKSPACE_SLUG}/chat"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"message": prompt_text, "mode": "chat", "settings": {"temperature": 0.9}}

    try:
        print(
            f"🚀 Generating batch {batch_num} (tasks for {len(combinations_chunk)} prompts)..."
        )
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json().get("textResponse", "").strip()
    except Exception as e:
        print(f"❌ Error during batch {batch_num}: {e}")
        return None


# --- 4. SYSTEMATIC MAIN LOOP WITH STRICT VALIDATION ---


if __name__ == "__main__":
    # 1. Generate all combinations mathematically
    all_combinations = list(
        itertools.product(
            FOCUS_TOPICS, JOURNEY_DESCRIPTIONS.keys(), PERSONA_DESCRIPTIONS.keys()
        )
    )

    # 2. Shuffle the list to distribute combinations evenly
    random.shuffle(all_combinations)

    TARGET_PROMPTS = len(all_combinations)  # Expected to be exactly 600
    BATCH_SIZE = 15  # Set to 15 to optimize language model performance

    with open(CSV_FILENAME, mode="w", newline="", encoding="utf-8-sig") as f:
        f.write('"User Prompt";"Focus Topic";"Journey Phase";"Persona"\n')

    print(
        f"🎬 Starting systematic generation. Exactly {TARGET_PROMPTS} unique combinations calculated."
    )

    recent_examples = []
    already_covered_combos = set()  # Set for exact tracking of generated combinations
    batch_counter = 1

    # Loop executes until all combinations are successfully processed
    while len(already_covered_combos) < TARGET_PROMPTS:
        # Identify missing combinations
        remaining_combos = [
            c for c in all_combinations if c not in already_covered_combos
        ]

        # Select the next batch
        chunk = remaining_combos[:BATCH_SIZE]

        raw_csv_text = generate_dataset_batch(batch_counter, chunk, recent_examples)

        if raw_csv_text:
            raw_csv_text = raw_csv_text.replace("```csv", "").replace("```", "").strip()
            lines = raw_csv_text.split("\n")

            valid_lines_in_batch = 0

            with open(CSV_FILENAME, mode="a", newline="", encoding="utf-8-sig") as f:
                for line in lines:
                    parts = line.split(";")
                    # Verify that the language model output contains exactly 4 columns
                    if len(parts) >= 4:
                        # Sanitize column values (remove whitespace and quotes)
                        u_prompt = parts[0].strip().strip('"')
                        u_topic = parts[1].strip().strip('"')
                        u_phase = parts[2].strip().strip('"')
                        u_persona = parts[3].strip().strip('"')

                        current_combo = (u_topic, u_phase, u_persona)

                        # VALIDATION MECHANISM:
                        # 1. Does this exactly match a combination requested in the current batch?
                        # 2. Is this combination missing? (Prevents duplicates)
                        if (
                            current_combo in chunk
                            and current_combo not in already_covered_combos
                        ):
                            # Format cleanly as CSV and append to file
                            f.write(
                                f'"{u_prompt}";"{u_topic}";"{u_phase}";"{u_persona}"\n'
                            )
                            already_covered_combos.add(current_combo)
                            valid_lines_in_batch += 1

                            # Update rolling memory of recent examples
                            if len(recent_examples) < 3:
                                recent_examples.append(u_prompt[:80] + "...")
                            else:
                                recent_examples.pop(0)
                                recent_examples.append(u_prompt[:80] + "...")

            print(
                f"✅ Batch {batch_counter}: {valid_lines_in_batch}/{len(chunk)} combinations verified. Progress: {len(already_covered_combos)}/{TARGET_PROMPTS}"
            )
        else:
            print(
                f"⚠️ Batch {batch_counter} failed. Retrying combinations in the next iteration..."
            )

        batch_counter += 1
        time.sleep(3)

    print(
        f"\n✨ Systematically completed: All {TARGET_PROMPTS} combinations converted to '{CSV_FILENAME}'."
    )
