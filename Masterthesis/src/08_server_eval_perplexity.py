"""Evaluates the perplexity of generated responses using a causal language model.

This module leverages a causal language model (Qwen1.5-1.8B) on a GPU to evaluate
the perplexity of texts. It systematically iterates through baseline scraped data
and data from various optimization methods, calculates the perplexity for each
source file, and aggregates the results into a CSV format.
"""

import csv
import glob
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    """Executes the server-side perplexity evaluation.

    The workflow encompasses the following steps:
    1. Locates the root directory to find scraped and optimized data.
    2. Initializes the causal language model and tokenizer on the GPU.
    3. Evaluates the perplexity of all original baseline web sources.
    4. Evaluates the perplexity of all optimized sources across optimization methods.
    5. Writes the aggregated results to a CSV output file.
    """
    # Look for the Masterthesis folder in the parent directory (like the local script)
    base_dir = "../Masterthesis"
    if not os.path.exists(base_dir):
        print(f"[Error] The directory {base_dir} was not found.")
        print(
            "Please ensure you have copied the entire 'Masterthesis' folder from your laptop to the server and are running this script inside the 'src' folder."
        )
        return

    model_id = "Qwen/Qwen1.5-1.8B"
    print(f"Loading model {model_id} into GPU memory (RTX 3090)...")

    # Loads the model onto the GPU (device_map="auto" automatically uses Cuda/GPU)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    device = model.device
    print(f"Model successfully loaded on Device: {device}")

    def get_ppl(text: str) -> float:
        """Calculates the perplexity of a given text.

        Args:
            text: The input text to evaluate.

        Returns:
            The calculated perplexity score as a float, or infinity if the text
            is empty or whitespace only.
        """
        if not text.strip():
            return float("inf")
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            return math.exp(loss.item())

    # Find all optimization methods that exist in the Masterthesis directory
    opt_dirs = glob.glob(os.path.join(base_dir, "optimized_data_*"))
    methods = [os.path.basename(d).replace("optimized_data_", "") for d in opt_dirs]

    results = []

    print("\nCalculating Perplexity for Baseline (Original Websites)...")
    prompt_dirs = glob.glob(os.path.join(base_dir, "scraped_data_full", "prompt_*"))
    for p_dir in prompt_dirs:
        prompt_id = os.path.basename(p_dir)
        for s_file in glob.glob(os.path.join(p_dir, "source_*.txt")):
            source_id = os.path.basename(s_file)
            with open(s_file, "r", encoding="utf-8") as f:
                text = f.read()
            ppl = get_ppl(text)
            results.append(
                {
                    "prompt": prompt_id,
                    "source": source_id,
                    "method": "baseline",
                    "perplexity": round(ppl, 4),
                }
            )

    # Evaluation of optimized methods
    for meth in methods:
        print(f"Calculating Perplexity for Method: {meth}...")
        prompt_dirs = glob.glob(
            os.path.join(base_dir, f"optimized_data_{meth}", "prompt_*")
        )
        for p_dir in prompt_dirs:
            prompt_id = os.path.basename(p_dir)
            for s_file in glob.glob(os.path.join(p_dir, "source_*.txt")):
                source_id = os.path.basename(s_file)
                with open(s_file, "r", encoding="utf-8") as f:
                    text = f.read()
                ppl = get_ppl(text)
                results.append(
                    {
                        "prompt": prompt_id,
                        "source": source_id,
                        "method": meth,
                        "perplexity": round(ppl, 4),
                    }
                )

    # Save results as CSV
    csv_file = os.path.join(base_dir, "perplexity_results_server.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(
            cf, fieldnames=["prompt", "source", "method", "perplexity"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n[SUCCESS] All calculations completed!")
    print(f"The results were saved in {csv_file}.")


if __name__ == "__main__":
    main()
