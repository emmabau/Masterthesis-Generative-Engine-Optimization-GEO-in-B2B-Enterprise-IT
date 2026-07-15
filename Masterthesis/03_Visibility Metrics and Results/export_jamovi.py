"""Script for exporting visibility metrics to a Jamovi-compatible dataset.

This script loads the processed dataset from the dashboard application, computes
Z-Scores for various visibility metrics relative to an 'Average Baseline',
pivots the data into a wide format, adds dummy variables for the different
Generative Engine Optimization (GEO) methods, and exports the final dataset
to a CSV file formatted for analysis in Jamovi.
"""

import sys

import numpy as np
import pandas as pd

# Add the dashboard directory to sys.path to import the app module
sys.path.append(r"C:\Users\finnb\Documents\emma\Masterthesis\Dashboard")
import app


def calculate_z_score(row, metrics_for_z, b_stats):
    """Calculate the Z-Score for a specific metric in a row.

    Args:
        row (pd.Series): A row from the dataset containing 'Metric' and 'Target_Score'.
        metrics_for_z (list): A list of metrics eligible for Z-Score calculation.
        b_stats (dict): A dictionary containing mean and standard deviation
            for the baseline methods.

    Returns:
        float: The calculated Z-Score, or NaN if the metric is invalid or
            if the standard deviation is zero.
    """
    metric = row["Metric"]
    if metric not in metrics_for_z:
        return np.nan

    b_mean = b_stats[metric]["mean"]
    b_std = b_stats[metric]["std"]
    val = row["Target_Score"]

    if pd.notnull(val) and pd.notnull(b_std) and b_std > 0:
        return (val - b_mean) / b_std

    return np.nan


def main():
    """Execute the data loading, Z-Score computation, and Jamovi export pipeline."""
    output_path = r"C:\Users\finnb\Documents\emma\Masterthesis\03_Visibility Metrics and Results\jamovi_dataset.csv"

    print("Loading data using app.load_data()...")
    df_merged, _, _, _, _ = app.load_data()

    print("Calculating Z-Scores...")
    metrics_for_z = [
        "Absolute Wordcount",
        "Position Adjusted Wordcount",
        "Chat Cite",
        "Sentence Cite",
        "GEval Overall",
        "ROUGE-L Recall",
        "BLEU Score",
        "RVS Sentiment",
        "Position Adjusted Sentence Count",
        "Perplexity",
    ]

    b_stats = {}
    df_b = df_merged[df_merged["Method"] == "Average Baseline"]
    for m in metrics_for_z:
        b_df_m = df_b[df_b["Metric"] == m]
        b_stats[m] = {
            "mean": b_df_m["Target_Score"].mean(),
            "std": b_df_m["Target_Score"].std(),
        }

    z_df = df_merged[df_merged["Metric"].isin(metrics_for_z)].copy()
    z_df["Z_Score"] = z_df.apply(
        lambda row: calculate_z_score(row, metrics_for_z, b_stats), axis=1
    )

    # Listwise deletion identical to Jamovi logic in app.py
    num_expected = len(metrics_for_z)
    valid_counts = z_df.groupby(["Prompt_ID", "Method"])["Target_Score"].count()
    valid_groups = valid_counts[valid_counts == num_expected].index
    z_df = z_df.set_index(["Prompt_ID", "Method"])
    z_df = z_df.loc[z_df.index.isin(valid_groups)].reset_index()

    # Pivot original Target_Score values to wide format
    print("Pivoting Target Scores...")
    wide_vals = (
        df_merged[df_merged["Metric"] != "Meta-Z-Score"]
        .pivot_table(
            index=["Prompt_ID", "Method", "Focus Topic", "Persona", "Journey Phase"],
            columns="Metric",
            values="Target_Score",
        )
        .reset_index()
    )

    # Pivot Z-Scores to wide format
    print("Pivoting Z-Scores...")
    wide_z = z_df.pivot_table(
        index=["Prompt_ID", "Method"], columns="Metric", values="Z_Score"
    ).reset_index()

    # Rename Z-Score columns to explicitly state '(Z-Score)'
    wide_z.columns = [
        col if col in ["Prompt_ID", "Method"] else f"{col} (Z-Score)"
        for col in wide_z.columns
    ]

    # Merge Target Scores and Z-Scores
    print("Merging data...")
    df_wide = pd.merge(wide_vals, wide_z, on=["Prompt_ID", "Method"], how="left")

    # Append Meta-Z-Score
    meta_d_df = df_merged[df_merged["Metric"] == "Meta-Z-Score"][
        ["Prompt_ID", "Method", "Target_Score"]
    ]
    meta_d_df = meta_d_df.rename(columns={"Target_Score": "Meta-Z-Score"})
    df_wide = pd.merge(df_wide, meta_d_df, on=["Prompt_ID", "Method"], how="left")

    # Add boolean Dummy variables for methods
    methods = sorted(df_merged["Method"].unique())
    for m in methods:
        df_wide[f"Methode_{m}"] = (df_wide["Method"] == m).astype(int)

    old_cols = [
        "Prompt_ID",
        "Method",
        "Focus Topic",
        "Persona",
        "Journey Phase",
        "Absolute Wordcount",
        "Absolute Wordcount (Z-Score)",
        "BLEU Score",
        "BLEU Score (Z-Score)",
        "Chat Cite",
        "Chat Cite (Z-Score)",
        "GEval Overall",
        "GEval Overall (Z-Score)",
        "Length Ratio",
        "Meta-Z-Score",
        "Perplexity",
        "Perplexity (Z-Score)",
        "Position Adjusted Sentence Count",
        "Position Adjusted Sentence Count (Z-Score)",
        "Position Adjusted Wordcount",
        "Position Adjusted Wordcount (Z-Score)",
        "ROUGE-L Recall",
        "ROUGE-L Recall (Z-Score)",
        "RVS Sentiment",
        "RVS Sentiment (Z-Score)",
        "Sentence Cite",
        "Sentence Cite (Z-Score)",
        "Methode_Authoritative",
        "Methode_AutoGEO",
        "Methode_Average Baseline",
        "Methode_Baseline",
        "Methode_Baseline V2",
        "Methode_Citing Credible Sources",
        "Methode_Fluent Optimization",
        "Methode_Inverted Pyramid",
        "Methode_LLMs.txt",
        "Methode_More Quotes",
        "Methode_Simple Language",
        "Methode_Statistics Optimization",
        "Methode_Technical Terms",
    ]

    # Ensure df_wide contains exactly the columns of old_cols in the specified order
    for col in old_cols:
        if col not in df_wide.columns:
            df_wide[col] = np.nan

    df_wide = df_wide[old_cols]

    print("Saving to jamovi_dataset.csv...")
    df_wide.to_csv(output_path, sep=";", index=False)
    print(f"Successfully exported {len(df_wide)} rows to {output_path}")


if __name__ == "__main__":
    main()
