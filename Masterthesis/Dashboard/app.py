"""Module for visualizing and analyzing visibility metrics for the Master Thesis.

This dashboard provides a Gradio-based interface to evaluate the impact of various
GEO (Generative Engine Optimization) strategies on LLM outputs. It computes stability
metrics, relative performance gains, and cannibalization effects, visualizing them
through dataframes and interactive plots.
"""

import re
import functools
import gradio as gr
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import os
import plotly.express as px

# Pfade zu deinen Masterthesis-Daten
ABSOLUTE_CSV_PATH = r"C:\Users\finnb\Documents\emma\Masterthesis\03_Visibility Metrics and Results\absolute_visibility_results.csv"
GEVAL_CSV_PATH = r"C:\Users\finnb\Documents\emma\Masterthesis\03_Visibility Metrics and Results\geval_absolute_results.csv"
STABILITY_CSV_PATH = r"C:\Users\finnb\Documents\emma\Masterthesis\03_Visibility Metrics and Results\stability_results.csv"
DATASET_CSV_PATH = r"C:\Users\finnb\Documents\emma\Masterthesis\01_Datapreparation\User Query creation\masterarbeit_dataset_systematisch.csv"


def load_data():
    """Load and preprocess absolute, stability, and evaluation datasets.

    Reads data from predefined absolute CSV paths, merges it with metadata and target
    indices, and calculates relative target and competitor scores. Adds artificial
    prompt IDs for accurate tag-wise filtering.

    Returns:
        tuple: A tuple containing:
            - df_merged (pd.DataFrame): Merged DataFrame with visibility metrics and metadata.
            - df_meta (pd.DataFrame): DataFrame containing metadata properties.
            - df_stability (pd.DataFrame): DataFrame containing stability metrics.
            - metrics_overall (list): Sorted list of unique values for overall metrics.
            - metrics_tag_wise (list): Sorted list of unique values for tag-wise metrics.
    """
    if not os.path.exists(ABSOLUTE_CSV_PATH) or not os.path.exists(DATASET_CSV_PATH):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], []

    df_abs = pd.read_csv(ABSOLUTE_CSV_PATH, sep=";")

    if os.path.exists(GEVAL_CSV_PATH):
        df_geval = pd.read_csv(GEVAL_CSV_PATH, sep=";")
        # Berechne den G-Eval Score als Ganzes (Durchschnitt der 7 Kategorien)
        df_geval_overall = (
            df_geval.groupby(["Prompt_ID", "Method"])[
                ["Source_0", "Source_1", "Source_2", "Source_3", "Source_4"]
            ]
            .mean()
            .reset_index()
        )
        df_geval_overall["Metric"] = "GEval Overall"
        # Füge nur den Gesamt-Score hinzu (die 7 Einzelkategorien werden weggelassen)
        df_abs = pd.concat([df_abs, df_geval_overall], ignore_index=True)

    # Load target indices
    TARGET_INDICES_PATH = r"C:\Users\finnb\Documents\emma\Masterthesis\03_Visibility Metrics and Results\target_indices.csv"
    if os.path.exists(TARGET_INDICES_PATH):
        df_target = pd.read_csv(TARGET_INDICES_PATH, sep=";")
        df_abs = pd.merge(df_abs, df_target, on="Prompt_ID", how="left")
        df_abs["Target_Index"] = df_abs["Target_Index"].fillna(0).astype(int)
    else:
        df_abs["Target_Index"] = 0

    df_meta = pd.read_csv(DATASET_CSV_PATH, sep=";")
    df_stability = pd.DataFrame()
    if os.path.exists(STABILITY_CSV_PATH):
        df_stability = pd.read_csv(STABILITY_CSV_PATH, sep=";")

    # Füge eine künstliche Prompt_ID zur Meta-Tabelle hinzu (1-basiert)
    df_meta["Prompt_ID"] = range(1, len(df_meta) + 1)
    # Merge, damit wir nach Tags filtern können
    df_merged = pd.merge(df_abs, df_meta, on="Prompt_ID", how="inner")

    # Calculate Target_Score for each row
    def get_target_score(row):
        """Extract the score of the target source for a given row.

        Args:
            row (pd.Series): A row from the merged DataFrame.

        Returns:
            float: The score associated with the designated target index.
        """
        idx = int(row["Target_Index"])
        col = f"Source_{idx}"
        return row[col] if pd.notnull(row[col]) else 0

    def get_comp_scores(row):
        """Retrieve the scores of competing sources for a given row.

        Args:
            row (pd.Series): A row from the merged DataFrame.

        Returns:
            pd.Series: A Series containing up to four competitor scores.
        """
        try:
            idx = int(row["Target_Index"])
            cols = [f"Source_{i}" for i in range(5) if i != idx]
            vals = [row[c] if pd.notnull(row[c]) else 0 for c in cols]
            while len(vals) < 4:
                vals.append(0)
            return pd.Series(vals[:4])
        except Exception:
            return pd.Series([0, 0, 0, 0])

    def get_other_score(row):
        """Calculate the average score of all competing sources.

        Args:
            row (pd.Series): A row from the merged DataFrame.

        Returns:
            float: The mean score of all sources excluding the target index.
        """
        try:
            idx = int(row["Target_Index"])
            cols = [f"Source_{i}" for i in range(5) if i != idx]
            vals = [row[c] for c in cols if pd.notnull(row[c])]
            return sum(vals) / len(vals) if vals else 0
        except Exception:
            return 0

    df_merged["Target_Score"] = df_merged.apply(get_target_score, axis=1)
    df_merged["Other_Score"] = df_merged.apply(get_other_score, axis=1)
    df_merged[["Comp_1_Score", "Comp_2_Score", "Comp_3_Score", "Comp_4_Score"]] = (
        df_merged.apply(get_comp_scores, axis=1)
    )

    # Metriken für Tag-Wise (alles was in df_abs ist)
    METRIC_LABELS = {
        "absolute_wordcount": "Absolute Wordcount",
        "position_adjusted_wordcount": "Position Adjusted Wordcount",
        "chat_cite": "Chat Cite",
        "sentence_cite": "Sentence Cite",
        "geval_overall": "GEval Overall",
        "rouge_l_recall": "ROUGE-L Recall",
        "bleu": "BLEU Score",
        "length_ratio": "Length Ratio",
        "rvs_sentiment": "RVS Sentiment",
        "perplexity_qwen": "Perplexity",
        "impression_pos_count_simple": "Position Adjusted Sentence Count",
    }

    METHOD_LABELS = {
        "Baseline_V2": "Baseline V2",
        "authoritative": "Authoritative",
        "autogeo_api_mine": "AutoGEO",
        "citing_credible": "Citing Credible Sources",
        "fluent_gpt": "Fluent Optimization",
        "inverted_pyramid_mine": "Inverted Pyramid",
        "llms_txt": "LLMs.txt",
        "more_quotes": "More Quotes",
        "simple_language": "Simple Language",
        "stats_optimization": "Statistics Optimization",
        "technical_terms": "Technical Terms",
        "identity": "Identity",
    }

    df_merged["Metric"] = df_merged["Metric"].replace(METRIC_LABELS)
    df_merged["Method"] = df_merged["Method"].replace(METHOD_LABELS)

    # Calculate actual Average Baseline
    numeric_cols = [
        "Source_0",
        "Source_1",
        "Source_2",
        "Source_3",
        "Source_4",
        "Target_Score",
        "Other_Score",
        "Comp_1_Score",
        "Comp_2_Score",
        "Comp_3_Score",
        "Comp_4_Score",
    ]
    df_base = df_merged[df_merged["Method"].isin(["Baseline", "Baseline V2"])]
    if not df_base.empty:
        df_avg_base = (
            df_base.groupby(["Prompt_ID", "Metric"])[numeric_cols].mean().reset_index()
        )
        df_avg_base["Method"] = "Average Baseline"

        # Merge metadata back
        _ = [
            c for c in df_merged.columns if c not in numeric_cols + ["Method", "Metric"]
        ]
        meta_df = df_base.drop_duplicates(subset=["Prompt_ID"]).drop(
            columns=numeric_cols + ["Method", "Metric"]
        )
        df_avg_base = pd.merge(df_avg_base, meta_df, on="Prompt_ID", how="left")

        # Append to df_merged
        df_merged = pd.concat([df_merged, df_avg_base], ignore_index=True)

    if not df_stability.empty:
        df_stability["Visibility Metric"] = df_stability["Visibility Metric"].replace(
            METRIC_LABELS
        )
        df_stability["Optimization Method"] = df_stability[
            "Optimization Method"
        ].replace(METHOD_LABELS)

    metrics_tag_wise = sorted(df_merged["Metric"].unique().tolist())

    metrics_overall = set(metrics_tag_wise)
    if not df_stability.empty:
        metrics_overall.update(df_stability["Visibility Metric"].unique().tolist())
    metrics_overall = sorted(list(metrics_overall))
    metrics_overall = [
        m for m in metrics_overall if m not in ["Length Ratio", "Perplexity"]
    ]

    metrics_for_z = [
        m for m in metrics_tag_wise if m not in ["Length Ratio", "Perplexity"]
    ]
    cols = [
        "Target_Score",
        "Other_Score",
        "Comp_1_Score",
        "Comp_2_Score",
        "Comp_3_Score",
        "Comp_4_Score",
        "Source_0",
        "Source_1",
        "Source_2",
        "Source_3",
        "Source_4",
    ]

    df_b = df_merged[df_merged["Method"] == "Average Baseline"]
    b_stats = {}
    for m in metrics_for_z:
        b_df_m = df_b[df_b["Metric"] == m]
        b_stats[m] = {
            col: {"mean": b_df_m[col].mean(), "std": b_df_m[col].std()} for col in cols
        }

    def calc_z(row, col):
        """Calculate the z-score for a specific column value based on baseline statistics.

        Args:
            row (pd.Series): A row containing metric values.
            col (str): The column name for which to compute the z-score.

        Returns:
            float: The computed z-score, or NaN if baseline deviation is zero or data is missing.
        """
        m = row["Metric"]
        if m not in metrics_for_z:
            return np.nan
        b_mean = b_stats[m][col]["mean"]
        b_std = b_stats[m][col]["std"]
        val = row[col]
        if pd.notnull(val) and pd.notnull(b_std) and b_std > 0:
            return (val - b_mean) / b_std
        return np.nan

    z_df = df_merged[df_merged["Metric"].isin(metrics_for_z)].copy()
    for col in cols:
        z_df[col] = z_df.apply(lambda row: calc_z(row, col), axis=1)

    # --- APPLY LISTWISE DELETION SIMILAR TO JAMOVI ---
    num_expected = len(metrics_for_z)
    valid_counts = z_df.groupby(["Prompt_ID", "Method"])["Target_Score"].count()
    valid_groups = valid_counts[valid_counts == num_expected].index

    z_df = z_df.set_index(["Prompt_ID", "Method"])
    z_df = z_df.loc[z_df.index.isin(valid_groups)].reset_index()
    # --------------------------------------------------

    meta_z_df = z_df.groupby(["Prompt_ID", "Method"])[cols].mean().reset_index()
    meta_z_df["Metric"] = "Meta-Z-Score"

    meta_attrs = df_meta.copy()
    meta_z_merged = pd.merge(meta_z_df, meta_attrs, on="Prompt_ID", how="inner")

    target_idx = df_merged[["Prompt_ID", "Method", "Target_Index"]].drop_duplicates()
    meta_z_merged = pd.merge(
        meta_z_merged, target_idx, on=["Prompt_ID", "Method"], how="inner"
    )

    df_merged = pd.concat([df_merged, meta_z_merged], ignore_index=True)

    metrics_tag_wise = ["Meta-Z-Score"] + metrics_tag_wise
    if "Meta-Z-Score" not in metrics_overall:
        metrics_overall = ["Meta-Z-Score"] + metrics_overall

    global global_wide_df
    try:
        global_wide_df = df_merged.pivot_table(
            index=["Prompt_ID", "Method", "Focus Topic", "Persona", "Journey Phase"],
            columns="Metric",
            values="Target_Score",
        ).reset_index()
    except Exception as e:
        print(f"Error creating wide df: {e}")
        global_wide_df = pd.DataFrame()

    return df_merged, df_meta, df_stability, metrics_overall, metrics_tag_wise


def get_adjusted_effects(df_wide, metric, focus_topics, journey_phases, personas):
    """Calculate baseline-adjusted effects for a specific metric using ANCOVA.

    Computes adjusted metric effects relative to the average baseline, controlling
    for focus topics, journey phases, and personas as covariates if they vary.

    Args:
        df_wide (pd.DataFrame): Wide-format DataFrame containing metric values.
        metric (str): The metric name to evaluate.
        focus_topics (list): List of focus topics to include.
        journey_phases (list): List of journey phases to include.
        personas (list): List of personas to include.

    Returns:
        dict: A dictionary mapping method names to their adjusted effect values.
            Returns None if the input DataFrame is insufficient or model fitting fails.
    """
    if df_wide is None or df_wide.empty:
        return None

    df_filtered = df_wide.copy()
    if focus_topics:
        df_filtered = df_filtered[df_filtered["Focus Topic"].isin(focus_topics)]
    if journey_phases:
        df_filtered = df_filtered[df_filtered["Journey Phase"].isin(journey_phases)]
    if personas:
        df_filtered = df_filtered[df_filtered["Persona"].isin(personas)]

    if len(df_filtered) < 10:
        return None

    def clean_name(name):
        """Sanitize a string for use as a variable name in statistical models.

        Args:
            name (str): The original string name.

        Returns:
            str: The sanitized name with non-alphanumeric characters replaced by underscores.
        """
        import re

        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    metric_clean = clean_name(metric)
    if metric not in df_filtered.columns:
        return None

    df_filtered[metric_clean] = df_filtered[metric]

    baseline_method = "Average Baseline"
    covariates = []
    if df_filtered["Focus Topic"].nunique() > 1:
        df_filtered["Focus_Topic"] = df_filtered["Focus Topic"]
        covariates.append("C(Focus_Topic, Sum)")
    if df_filtered["Persona"].nunique() > 1:
        covariates.append("C(Persona, Sum)")
    if df_filtered["Journey Phase"].nunique() > 1:
        df_filtered["Journey_Phase"] = df_filtered["Journey Phase"]
        covariates.append("C(Journey_Phase, Sum)")

    if (
        metric != "Perplexity"
        and "Perplexity" in df_filtered.columns
        and df_filtered["Perplexity"].notna().any()
    ):
        df_filtered["Perplexity_c"] = (
            df_filtered["Perplexity"] - df_filtered["Perplexity"].mean()
        )
        covariates.append("Perplexity_c")
    if (
        metric != "Length Ratio"
        and "Length Ratio" in df_filtered.columns
        and df_filtered["Length Ratio"].notna().any()
    ):
        df_filtered["Length_Ratio_c"] = (
            df_filtered["Length Ratio"] - df_filtered["Length Ratio"].mean()
        )
        covariates.append("Length_Ratio_c")

    formula = f"{metric_clean} ~ C(Method, Treatment('{baseline_method}'))"
    if covariates:
        formula += " + " + " + ".join(covariates)

    try:
        mod = smf.ols(formula, data=df_filtered).fit()

        adj_effects = {}
        intercept = mod.params["Intercept"]
        method_params = mod.params.filter(like="C(Method")

        is_meta = metric in ["Meta-Z-Score", "Meta-Z-Score"]
        if is_meta:
            adj_effects[baseline_method] = intercept

        for idx, val in method_params.items():
            match = re.search(r"\[T\.(.+?)\]", idx)
            if match:
                method_name = match.group(1)
                if is_meta:
                    adj_effects[method_name] = intercept + val
                else:
                    adj_effects[method_name] = val

        return adj_effects
    except Exception as e:
        print(f"Regression failed for {metric}: {e}")
        return None





@functools.lru_cache(maxsize=128)
def _cached_calculate_leaderboard(metric, focus_tuple, journey_tuple, persona_tuple):
    """Compute and format the stability and performance leaderboard, cached for efficiency.

    Calculates aggregate metrics such as WCP, DR, and WTR, applying ANCOVA adjustments
    where applicable. The function builds a comparison DataFrame for the selected metric.

    Args:
        metric (str): The selected visibility metric.
        focus_tuple (tuple): Tuple of selected focus topics.
        journey_tuple (tuple): Tuple of selected journey phases.
        persona_tuple (tuple): Tuple of selected personas.

    Returns:
        pd.DataFrame: Formatted leaderboard DataFrame sorted by relative improvement.
    """
    global global_wide_df
    if "global_wide_df" not in globals() or global_wide_df.empty:
        return pd.DataFrame()

    df_wide = global_wide_df.copy()
    if focus_tuple:
        df_wide = df_wide[df_wide["Focus Topic"].isin(focus_tuple)]
    if journey_tuple:
        df_wide = df_wide[df_wide["Journey Phase"].isin(journey_tuple)]
    if persona_tuple:
        df_wide = df_wide[df_wide["Persona"].isin(persona_tuple)]

    # We must construct a long df equivalent for WCP, DR etc
    df_m = df_wide.melt(
        id_vars=["Prompt_ID", "Method", "Focus Topic", "Persona", "Journey Phase"],
        value_vars=[metric],
        value_name="Target_Score",
    )
    df_m = df_m.dropna(subset=["Target_Score"])

    is_pct = metric in ["Chat Cite", "ROUGE-L Recall", "BLEU Score"]
    is_zscore = metric == "Meta-Z-Score"

    col_name = (
        "Sichtbarkeitsverschiebung (%-Punkte)"
        if is_pct
        else (
            "Sichtbarkeitsverschiebung (Z-Score)"
            if is_zscore
            else "Sichtbarkeitsverschiebung (%)"
        )
    )

    if df_m.empty:
        return pd.DataFrame(
            columns=[
                "GEO-Methode",
                col_name,
                "Durchschnittliches Delta",
                "Effektstärke (Z-Score)",
                "Verbesserungsrate (WCP %)",
                "Verschlechterungsrate (DR %)",
                "Nutzen-Risiko-Verhltnis (WTR)",
            ]
        )

    baseline_df = df_m[df_m["Method"] == "Average Baseline"][
        ["Prompt_ID", "Target_Score"]
    ].set_index("Prompt_ID")
    baseline_df.rename(columns={"Target_Score": "Baseline_Score"}, inplace=True)

    methods = [m for m in df_m["Method"].unique() if m != "Average Baseline"]
    if not methods:
        return pd.DataFrame()

    # We merge all methods with baseline
    comp_list = []
    for method in methods:
        method_df = df_m[df_m["Method"] == method][
            ["Prompt_ID", "Target_Score"]
        ].set_index("Prompt_ID")
        comp = method_df.join(baseline_df, how="inner")
        comp["Method"] = method
        comp_list.append(comp)

    if not comp_list:
        return pd.DataFrame()

    comp = pd.concat(comp_list)
    comp["Delta"] = comp["Target_Score"] - comp["Baseline_Score"]
    baseline_std = comp["Baseline_Score"].std()
    avg_baseline = comp["Baseline_Score"].mean()

    agg_df = (
        comp.groupby("Method")
        .agg(
            avg_delta=("Delta", "mean"),
            avg_method=("Target_Score", "mean"),
            wcp=("Delta", lambda x: (x > 0).mean() * 100),
            dr=("Delta", lambda x: (x < 0).mean() * 100),
        )
        .reset_index()
    )

    # --- ANCOVA ADJUSTMENT ---
    adj_effects = get_adjusted_effects(
        global_wide_df, metric, focus_tuple, journey_tuple, persona_tuple
    )
    if adj_effects:
        agg_df["avg_delta"] = agg_df.apply(
            lambda row: adj_effects.get(row["Method"], row["avg_delta"]), axis=1
        )
    # -------------------------

    agg_df["wtr"] = agg_df.apply(
        lambda row: (
            (row["wcp"] / row["dr"])
            if row["dr"] > 0
            else (row["wcp"] if row["wcp"] > 0 else 0)
        ),
        axis=1,
    )

    agg_df["z_score"] = 0.0
    if pd.notnull(baseline_std) and baseline_std > 0:
        if is_zscore:
            agg_df["z_score"] = agg_df["avg_delta"]
        else:
            agg_df["z_score"] = agg_df["avg_delta"] / baseline_std

    if is_zscore:
        agg_df["verschiebung"] = agg_df["avg_delta"]
    elif is_pct:
        agg_df["verschiebung"] = agg_df["avg_delta"] * 100
    else:
        agg_df["verschiebung"] = agg_df.apply(
            lambda row: (
                (row["avg_delta"] / avg_baseline * 100) if avg_baseline != 0 else 0
            ),
            axis=1,
        )

    agg_df.rename(
        columns={
            "Method": "GEO-Methode",
            "verschiebung": col_name,
            "avg_delta": "Durchschnittliches Delta",
            "z_score": "Effektstärke (Z-Score)",
            "wcp": "Verbesserungsrate (WCP %)",
            "dr": "Verschlechterungsrate (DR %)",
            "wtr": "Nutzen-Risiko-Verhltnis (WTR)",
        },
        inplace=True,
    )

    res_df = agg_df[
        [
            "GEO-Methode",
            col_name,
            "Durchschnittliches Delta",
            "Effektstärke (Z-Score)",
            "Verbesserungsrate (WCP %)",
            "Verschlechterungsrate (DR %)",
            "Nutzen-Risiko-Verhltnis (WTR)",
        ]
    ]

    res_df = res_df.round(
        {
            col_name: 5,
            "Durchschnittliches Delta": 5,
            "Effektstärke (Z-Score)": 5,
            "Verbesserungsrate (WCP %)": 2,
            "Verschlechterungsrate (DR %)": 2,
            "Nutzen-Risiko-Verhltnis (WTR)": 2,
        }
    )
    res_df = res_df.sort_values(by=col_name, ascending=False).reset_index(drop=True)
    return res_df


def calculate_leaderboard(
    df_merged, selected_metric, focus_topics, journey_phases, personas
):
    """Wrapper function to compute the leaderboard using cached results.

    Args:
        df_merged (pd.DataFrame): The main merged DataFrame (unused directly, kept for API consistency).
        selected_metric (str): The metric to evaluate.
        focus_topics (list): List of focus topics to include.
        journey_phases (list): List of journey phases to include.
        personas (list): List of personas to include.

    Returns:
        pd.DataFrame: The computed leaderboard.
    """
    return _cached_calculate_leaderboard(
        selected_metric,
        tuple(sorted(focus_topics)),
        tuple(sorted(journey_phases)),
        tuple(sorted(personas)),
    )


def get_diff_html_words(base_text, opt_text, max_words=300):
    """Generate an HTML-formatted visual diff between two text strings.

    Highlights inserted and replaced text in green, and deleted text in red.
    Unchanged segments longer than a threshold are collapsible to save space.

    Args:
        base_text (str): The original baseline text.
        opt_text (str): The optimized text.
        max_words (int, optional): The maximum number of visible words before truncation. Defaults to 300.

    Returns:
        tuple: A tuple containing:
            - str: The generated HTML string representing the diff.
            - bool: True if changes were detected, False otherwise.
    """
    from difflib import SequenceMatcher
    import html

    base_words = base_text.split()
    opt_words = opt_text.split()

    matcher = SequenceMatcher(None, base_words, opt_words)
    diff_html = []

    has_changes = False
    word_count = 0
    is_hidden = False

    def append_html_block(words, tag_type):
        """Append a stylized HTML span block representing a diff segment.

        Args:
            words (list): List of words in the segment.
            tag_type (str): The type of diff tag ('insert', 'delete', 'replace', or 'equal').
        """
        if not words:
            return
        text = html.escape(" ".join(words))
        if tag_type in ("insert", "replace_ins"):
            diff_html.append(
                f"<span style='background-color: #d4edda; color: #155724; padding: 2px; border-radius: 3px; font-weight: bold;'>{text}</span>"
            )
        elif tag_type in ("delete", "replace_del"):
            diff_html.append(
                f"<span style='background-color: #f8d7da; color: #721c24; padding: 2px; text-decoration: line-through; border-radius: 3px;'>{text}</span>"
            )
        elif tag_type == "equal":
            diff_html.append(text)

    def process_block(words, tag_type):
        """Process a block of words and manage text truncation and toggles.

        Args:
            words (list): List of words in the diff segment.
            tag_type (str): The type of diff operation.
        """
        nonlocal word_count, is_hidden
        if not words:
            return

        if is_hidden:
            append_html_block(words, tag_type)
            word_count += len(words)
            return

        if word_count + len(words) <= max_words:
            append_html_block(words, tag_type)
            word_count += len(words)
            return

        split_idx = max_words - word_count
        if split_idx > 0:
            append_html_block(words[:split_idx], tag_type)

        diff_html.append(
            "<details style='margin-top: 10px; cursor: pointer;'><summary style='color: #007bff; font-weight: bold; padding: 5px 0;'>... (Text gekürzt - hier klicken, um den Rest aufzuklappen)</summary><div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background: #fafafa; margin-top: 5px;'>"
        )
        is_hidden = True

        if split_idx < len(words):
            append_html_block(words[split_idx:], tag_type)

        word_count += len(words)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            has_changes = True

        if tag == "insert":
            process_block(opt_words[j1:j2], "insert")
        elif tag == "replace":
            process_block(base_words[i1:i2], "replace_del")
            process_block(opt_words[j1:j2], "replace_ins")
        elif tag == "delete":
            process_block(base_words[i1:i2], "delete")
        elif tag == "equal":
            chunk = base_words[i1:i2]
            if not has_changes and len(chunk) > 30:
                hidden_chunk = chunk[:-10]
                visible_chunk = chunk[-10:]
                diff_html.append(
                    "<details style='display: inline;'><summary style='color: gray; display: inline-block; cursor: pointer;'><em>... (Unveränderter Text ausgeblendet) ...</em></summary><span style='color: gray;'> "
                    + html.escape(" ".join(hidden_chunk))
                    + " </span></details> "
                )
                process_block(visible_chunk, "equal")
            elif len(chunk) > 40:
                process_block(chunk[:15], "equal")
                hidden_chunk = chunk[15:-15]
                diff_html.append(
                    " <details style='display: inline;'><summary style='color: gray; display: inline-block; cursor: pointer;'><em>... (Unveränderter Text ausgeblendet) ...</em></summary><span style='color: gray;'> "
                    + html.escape(" ".join(hidden_chunk))
                    + " </span></details> "
                )
                process_block(chunk[-15:], "equal")
            else:
                process_block(chunk, "equal")

    if is_hidden:
        diff_html.append("</div></details>")

    return " ".join(diff_html), has_changes


def _get_example_row(method, df_m, df_b, mode, metric):
    """Generate an HTML table row demonstrating text differences for a specific GEO method.

    Selects a representative, best, or worst prompt for the given method by comparing
    target scores against the baseline. Opens local text files to compute a word-level diff.

    Args:
        method (str): The name of the GEO optimization method.
        df_m (pd.DataFrame): DataFrame filtered to a specific metric.
        df_b (pd.DataFrame): DataFrame containing query metadata and original queries.
        mode (str): Evaluation mode ('representative', 'top', or 'worst').
        metric (str): The metric name to display.

    Returns:
        str: An HTML string representing a single table row with the formatted diff and scores.
    """
    import os
    import pandas as pd

    base_opt = r"C:\Users\finnb\Documents\emma\Masterthesis\02_Benchmark\AI Answers - with GEO optimization"
    base_base = r"C:\Users\finnb\Documents\emma\Masterthesis\01_Datapreparation\Webcrawling of User Queries\scraped_data_full"

    dir_map = {
        "Average Baseline": "Baseline",
        "Authoritative": "optimized_data_authoritative",
        "AutoGEO": "optimized_data_autogeo_api",
        "Citing Credible Sources": "optimized_data_citing_credible",
        "Fluent Optimization": "optimized_data_fluent_gpt",
        "Inverted Pyramid": "optimized_data_inverted_pyramid_mine",
        "LLMs.txt": "optimized_data_llms_txt",
        "Simple Language": "optimized_data_simple_language",
        "More Quotes": "optimized_data_more_quotes",
        "Statistics Optimization": "optimized_data_stats_optimization",
        "Technical Terms": "optimized_data_technical_terms",
    }
    method_dir = dir_map.get(method, method)

    comp_df = df_m[df_m["Method"] == method][
        ["Prompt_ID", "Target_Score", "Target_Index"]
    ].set_index("Prompt_ID")
    base_scores = df_m[df_m["Method"] == "Average Baseline"][
        ["Prompt_ID", "Target_Score"]
    ].set_index("Prompt_ID")
    comp = comp_df.join(base_scores, lsuffix="_meth", rsuffix="_base", how="inner")

    if comp.empty:
        return ""

    comp["Diff"] = comp["Target_Score_meth"] - comp["Target_Score_base"]

    if mode == "representative":
        mean_diff = comp["Diff"].mean()
        comp["Sort_Val"] = (comp["Diff"] - mean_diff).abs()
        comp_sorted = comp.sort_values(by="Sort_Val")
    elif mode == "top":
        comp_sorted = comp.sort_values(by="Diff", ascending=False)
    else:  # worst
        comp_sorted = comp.sort_values(by="Diff", ascending=True)

    best_diff_html = "<em>Keine sichtbaren Text-Änderungen (Diff leer)</em>"
    best_prompt_id = comp_sorted.index[0]  # Fallback to top if no diff found

    for prompt_id in comp_sorted.index:
        prompt_str = f"prompt_{int(prompt_id):03d}"
        target_idx = comp.loc[prompt_id, "Target_Index"]
        if pd.isna(target_idx):
            target_idx = 0

        target_file_idx = int(target_idx) + 1

        opt_path = os.path.join(
            base_opt, method_dir, prompt_str, f"source_{target_file_idx}.txt"
        )
        base_path = os.path.join(base_base, prompt_str, f"source_{target_file_idx}.txt")

        if os.path.exists(opt_path) and os.path.exists(base_path):
            try:
                with open(opt_path, "r", encoding="utf-8") as f_opt:
                    opt_text = (
                        f_opt.read()
                        .replace("<Output>", "")
                        .replace("</Output>", "")
                        .strip()
                    )
                with open(base_path, "r", encoding="utf-8") as f_base:
                    base_text = (
                        f_base.read()
                        .replace("<Output>", "")
                        .replace("</Output>", "")
                        .strip()
                    )

                # Skip cases where the text was completely rewritten (less than 15% vocabulary overlap)
                # or completely deleted (too short)
                opt_words = opt_text.split()
                base_words = base_text.split()

                if len(opt_words) < 20 and len(base_words) > 50:
                    continue

                set_base = set(base_words)
                set_opt = set(opt_words)
                union_len = len(set_base.union(set_opt))
                if union_len > 0:
                    jaccard = len(set_base.intersection(set_opt)) / union_len
                    # Require at least 25% vocabulary overlap to ensure the text wasn't completely replaced/deleted
                    if jaccard < 0.25:
                        continue

                diff_html, has_changes = get_diff_html_words(
                    base_text, opt_text, max_words=300
                )

                if has_changes:
                    best_diff_html = diff_html
                    best_prompt_id = prompt_id
                    
                    break
            except Exception:
                pass

    # best_prompt_id is now either the best with diff, or the absolute mathematically best without diff
    q = "Unbekannt"
    if not df_b.empty and "id" in df_b.columns:
        q_rows = df_b[df_b["id"] == int(best_prompt_id)]["query"]
        if q_rows.empty:
            q_rows = df_b[df_b["id"] == str(best_prompt_id)]["query"]
        if not q_rows.empty:
            q = q_rows.iloc[0]

    det_diff = comp.loc[best_prompt_id, "Diff"]
    det_m = comp.loc[best_prompt_id, "Target_Score_meth"]
    det_b = comp.loc[best_prompt_id, "Target_Score_base"]
    color = "green" if det_diff > 0 else ("red" if det_diff < 0 else "black")
    best_rel_imp = f"<span style='font-size: 1.1em; font-weight: bold; color: {color};'>{det_diff:+.2f}</span><br><span style='font-size: 0.85em; color: gray;'>Base: {det_b:.2f}<br>Meth: {det_m:.2f}</span>"

    mode_tag = ""
    if mode == "top":
        mode_tag = (
            "<br><span style='font-size: 0.8em; color: gold;'>⭐ Top-Performer</span>"
        )
    if mode == "worst":
        mode_tag = (
            "<br><span style='font-size: 0.8em; color: red;'>📉 Worst-Performer</span>"
        )

    return f"<tr><td><strong>{method}</strong>{mode_tag}</td><td>{best_prompt_id}</td><td><div style='font-size: 0.9em; line-height: 1.4;'><strong>Query:</strong> <em>{q}</em><br><br>{best_diff_html}</div></td><td style='text-align: center; vertical-align: middle;'>{best_rel_imp}</td></tr>"


def generate_examples_table_mode(metric, df_merged, mode):
    """Generate an HTML table showcasing text examples for various methods.

    Iterates through all non-baseline GEO methods to gather their corresponding
    example rows and compiles them into a styled HTML table.

    Args:
        metric (str): The metric being evaluated.
        df_merged (pd.DataFrame): The merged DataFrame with all score data.
        mode (str): The evaluation mode ('representative', 'top', or 'worst').

    Returns:
        str: An HTML string representing the complete examples table.
    """
    if df_merged.empty or not metric:
        return "<p>Keine Daten verfügbar.</p>"

    df_m = df_merged[df_merged["Metric"] == metric]

    methods = [
        m
        for m in df_merged["Method"].unique()
        if m not in ["Average Baseline", "Baseline", "Baseline V2"]
    ]

    html = f"<table class='table table-striped' border='1' style='width: 100%; text-align: left;'><thead><tr><th style='width: 15%;'>GEO-Methode</th><th style='width: 5%;'>Prompt ID</th><th>Text-Anpassung (Diff vs. Baseline)</th><th style='width: 15%; text-align: center;'>Effekt bei diesem Prompt<br>({metric})</th></tr></thead><tbody>"

    import os
    import pandas as pd

    csv_path = r"C:\Users\finnb\Documents\emma\Masterthesis\01_Datapreparation\Webcrawling of User Queries\my_geo_bench.csv"
    df_b = pd.DataFrame()
    if os.path.exists(csv_path):
        df_b = pd.read_csv(csv_path)

    for method in methods:
        html += _get_example_row(method, df_m, df_b, mode, metric)

    html += "</tbody></table>"
    return f"<div style='overflow-x: auto;'>{html}</div>"


def generate_examples_html(metric, df_merged):
    """Generate an HTML table with representative examples for each method.

    Args:
        metric (str): The metric being evaluated.
        df_merged (pd.DataFrame): The main merged DataFrame.

    Returns:
        str: An HTML string representing the representative examples table.
    """
    return generate_examples_table_mode(metric, df_merged, "representative")


def generate_top_performer_html(metric, df_merged):
    """Generate an HTML table displaying the best performing examples for each method.

    Args:
        metric (str): The metric being evaluated.
        df_merged (pd.DataFrame): The main merged DataFrame.

    Returns:
        str: An HTML string representing the top performer examples table.
    """
    return generate_examples_table_mode(metric, df_merged, "top")


def generate_worst_performer_html(metric, df_merged):
    """Generate an HTML table displaying the worst performing examples for each method.

    Args:
        metric (str): The metric being evaluated.
        df_merged (pd.DataFrame): The main merged DataFrame.

    Returns:
        str: An HTML string representing the worst performer examples table.
    """
    return generate_examples_table_mode(metric, df_merged, "worst")


# Erstelle die Gradio App
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Masterthesis: Generative Engine Optimization im B2B-IT-Kontext")

    df_merged, df_meta, df_stability, available_metrics, metrics_tag_wise = load_data()

    if df_merged.empty:
        gr.Warning(
            "Daten nicht gefunden! Stelle sicher, dass die CSV-Dateien existieren."
        )
    else:
        metric_init_overall = (
            "Meta-Z-Score"
            if "Meta-Z-Score" in available_metrics
            else (available_metrics[0] if available_metrics else None)
        )

        def update_overall(metric, focus, journey, persona):
            """Update all overall UI components based on selected filters.

            Args:
                metric (str): The currently selected evaluation metric.
                focus (list): Selected focus topics.
                journey (list): Selected journey phases.
                persona (list): Selected personas.

            Returns:
                tuple: Updated components including the leaderboard DataFrame,
                    the downloadable CSV path, HTML tables for tags, and plots.
            """
            if not metric:
                return pd.DataFrame(), None, "", "", "", None, None

            df_res = calculate_leaderboard(df_merged, metric, focus, journey, persona)

            def calculate_cross_tag_leaderboard(m, tag_col):
                """Generate a cross-tabulated leaderboard for a specific metadata tag.

                Args:
                    m (str): The metric to evaluate.
                    tag_col (str): The metadata column to group by (e.g., 'Focus Topic').

                Returns:
                    tuple: A tuple containing:
                        - str: HTML representation of the cross-tabulated table.
                        - go.Figure: A Plotly Heatmap figure visualizing performance across tags.
                """
                if df_merged.empty or tag_col not in df_merged.columns:
                    return "<p>Keine Tag-Aufschlüsselung verfügbar.</p>"

                # Sort methods by their overall leaderboard performance
                overall_res = calculate_leaderboard(df_merged, m, [], [], [])
                if not overall_res.empty:
                    methods = [
                        method
                        for method in overall_res["GEO-Methode"]
                        if method not in ["Baseline", "Baseline V2", "Average Baseline"]
                    ]
                else:
                    methods = sorted(
                        [
                            meth
                            for meth in df_merged["Method"].unique()
                            if meth
                            not in ["Baseline", "Baseline V2", "Average Baseline"]
                        ]
                    )

                tags = sorted(df_merged[tag_col].dropna().unique())

                results = []
                for method in methods:
                    row_res = {"GEO-Methode": method}

                    # Add Overall (Gesamt) Score first
                    if (
                        not overall_res.empty
                        and method in overall_res["GEO-Methode"].values
                    ):
                        score_col_overall = overall_res.columns[1]
                        row_res["Gesamt"] = overall_res[
                            overall_res["GEO-Methode"] == method
                        ].iloc[0][score_col_overall]
                    else:
                        row_res["Gesamt"] = None

                    for tag in tags:
                        # We just use calculate_leaderboard for each tag to get the exact value
                        tag_res = calculate_leaderboard(
                            df_merged,
                            m,
                            [tag] if tag_col == "Focus Topic" else [],
                            [tag] if tag_col == "Journey Phase" else [],
                            [tag] if tag_col == "Persona" else [],
                        )
                        if (
                            not tag_res.empty
                            and method in tag_res["GEO-Methode"].values
                        ):
                            # Get the score (column 1)
                            score_col = tag_res.columns[1]
                            val = tag_res[tag_res["GEO-Methode"] == method].iloc[0][
                                score_col
                            ]
                            row_res[tag] = val
                        else:
                            row_res[tag] = None
                    results.append(row_res)
                res_df = pd.DataFrame(results)
                if not res_df.empty:
                    res_df = res_df.set_index("GEO-Methode")
                    html_code = res_df.to_html(
                        border=1, classes="table table-striped", na_rep="-"
                    )
                    # Create Heatmap
                    import plotly.express as px

                    # We replace non-numeric with None or handle dynamically
                    num_df = res_df.apply(pd.to_numeric, errors="coerce")

                    # Calculate Delta relative to 'Gesamt' for coloring
                    delta_df = num_df.subtract(num_df["Gesamt"], axis=0)

                    # Keep GEO-Methods on the Y-axis and Tags on the X-axis
                    fig = px.imshow(
                        delta_df,
                        aspect="auto",
                        color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0,
                        title=f"Performance Heatmap (Color = Abweichung vom Gesamtwert): {tag_col}",
                    )
                    # Overlay the absolute values as text
                    fig.update_traces(text=num_df.values, texttemplate="%{text:.2f}")
                    # Update layout to ensure labels are visible and legible
                    fig.update_layout(xaxis_title=tag_col, yaxis_title="GEO-Methode")
                    return (
                        f'<div style="overflow-x: auto; white-space: nowrap;">{html_code}</div>',
                        fig,
                    )
                return "<p>Keine Daten.</p>", None

            tmp_path = f"overall_{metric}.csv" if metric else "overall_None.csv"
            df_res.to_csv(tmp_path, sep=";", index=False)
            html_table = (
                df_res.to_html(index=False, border=1)
                if not df_res.empty
                else "<p>Keine Daten verfügbar.</p>"
            )

            topic_html, topic_fig = calculate_cross_tag_leaderboard(
                metric, "Focus Topic"
            )
            persona_html, persona_fig = calculate_cross_tag_leaderboard(
                metric, "Persona"
            )
            journey_html, journey_fig = calculate_cross_tag_leaderboard(
                metric, "Journey Phase"
            )

            fig = None
            fig_z = None
            if not df_res.empty:
                col_to_plot = df_res.columns[1]
                if col_to_plot in df_res.columns:
                    fig = px.bar(
                        df_res,
                        x="GEO-Methode",
                        y=col_to_plot,
                        color="GEO-Methode",
                        text=col_to_plot,
                        title=f"Overall Leaderboard: {metric}",
                    )
                if "Effektstärke (Z-Score)" in df_res.columns:
                    fig_z = px.bar(
                        df_res,
                        x="GEO-Methode",
                        y="Effektstärke (Z-Score)",
                        color="GEO-Methode",
                        text="Effektstärke (Z-Score)",
                        title=f"Effektstärke (Z-Score): {metric}",
                    )
                    fig_z.add_hline(y=0, line_dash="dash", line_color="black")

            return (
                html_table,
                tmp_path,
                topic_html,
                persona_html,
                journey_html,
                fig,
                topic_fig,
                persona_fig,
                journey_fig,
            )

        if metric_init_overall:
            (
                html_init_overall,
                path_init_overall,
                init_topic,
                init_persona,
                init_journey,
                init_fig,
                init_topic_fig,
                init_persona_fig,
                init_journey_fig,
            ) = update_overall(metric_init_overall, [], [], [])
        else:
            (
                html_init_overall,
                path_init_overall,
                init_topic,
                init_persona,
                init_journey,
                init_fig,
                init_topic_fig,
                init_persona_fig,
                init_journey_fig,
            ) = "", None, "", "", "", None, None, None, None

        with gr.Column():
            with gr.Accordion(
                "1. Erstellung einer durchschnittlichen Baseline", open=False
            ):
                gr.Markdown(
                    "Dieser Abschnitt vergleicht den Effekt der GEO-Methoden gegen zwei verschiedene Baselines: Die initiale `Baseline` und die aggregierte `Average Baseline` (V2)."
                )

                def generate_baseline_bar_chart():
                    """Generate a Plotly Bar chart comparing Base1 to Base2 across selected metrics.

                    Returns:
                        go.Figure: A bar chart visualization showing absolute baseline values.
                    """
                    if df_merged.empty:
                        return None
                    metrics = [
                        m for m in df_merged["Metric"].unique() if m != "Meta-Z-Score"
                    ]

                    x_vals = []
                    y_vals = []
                    text_vals = []
                    colors = []

                    for metric in metrics:
                        is_pct = metric in ["Chat Cite", "ROUGE-L Recall", "BLEU Score"]
                        df_m = df_merged[df_merged["Metric"] == metric]

                        avg_b1 = df_m[df_m["Method"] == "Baseline"][
                            "Target_Score"
                        ].mean()
                        avg_b2 = df_m[df_m["Method"] == "Baseline V2"][
                            "Target_Score"
                        ].mean()

                        if pd.isna(avg_b1):
                            avg_b1 = 0
                        if pd.isna(avg_b2):
                            avg_b2 = 0

                        if is_pct:
                            delta = (avg_b2 - avg_b1) * 100
                            unit = "%-Pkte"
                        else:
                            delta = (
                                ((avg_b2 - avg_b1) / avg_b1 * 100)
                                if avg_b1 and avg_b1 != 0
                                else 0
                            )
                            unit = "%"

                        x_vals.append(metric)
                        y_vals.append(delta)
                        text_vals.append(f"{delta:+.2f} {unit}")
                        colors.append("positiv" if delta >= 0 else "negativ")

                    df_bar = pd.DataFrame(
                        {
                            "Metrik": x_vals,
                            "Abweichung": y_vals,
                            "Text": text_vals,
                            "Farbe": colors,
                        }
                    )

                    fig = px.bar(
                        df_bar,
                        x="Metrik",
                        y="Abweichung",
                        text="Text",
                        color="Farbe",
                        color_discrete_map={"positiv": "#2ca02c", "negativ": "#d62728"},
                        title="Gesamte Suchmaschinen-Schwankung: Baseline V2 vs. Baseline V1",
                    )
                    fig.update_traces(textposition="outside")
                    fig.update_layout(showlegend=False)
                    return fig

                baseline_bar = gr.Plot(value=generate_baseline_bar_chart())

                def generate_heatmap_baseline():
                    """Generate a Plotly Heatmap comparing improvements against Base1 and Base2.

                    Visualizes the discrepancy (Delta) between improvements measured
                    against two different baselines (Baseline vs Average Baseline).

                    Returns:
                        go.Figure: A Heatmap figure representing baseline measurement drift.
                    """
                    if df_merged.empty:
                        return None
                    metrics = sorted(
                        [m for m in df_merged["Metric"].unique() if m != "Meta-Z-Score"]
                    )
                    methods = sorted(
                        [
                            m
                            for m in df_merged["Method"].unique()
                            if m not in ["Baseline", "Baseline V2", "Average Baseline"]
                        ]
                    )
                    if not methods or not metrics:
                        return None

                    data = []
                    text = []
                    for method in methods:
                        row_data = []
                        row_text = []
                        for metric in metrics:
                            is_pct = metric in [
                                "Chat Cite",
                                "ROUGE-L Recall",
                                "BLEU Score",
                            ]
                            is_zscore = metric == "Meta-Z-Score"
                            df_m = df_merged[df_merged["Metric"] == metric]

                            avg_m = df_m[df_m["Method"] == method][
                                "Target_Score"
                            ].mean()
                            avg_b1 = df_m[df_m["Method"] == "Baseline"][
                                "Target_Score"
                            ].mean()
                            avg_b2 = df_m[df_m["Method"] == "Average Baseline"][
                                "Target_Score"
                            ].mean()

                            if pd.isna(avg_m):
                                avg_m = 0
                            if pd.isna(avg_b1):
                                avg_b1 = 0
                            if pd.isna(avg_b2):
                                avg_b2 = 0

                            def calc_imp(a_m, a_b):
                                """Calculate the relative or absolute improvement over the baseline.

                                Args:
                                    a_m (float): The mean score of the optimization method.
                                    a_b (float): The mean score of the baseline.

                                Returns:
                                    float: The computed improvement value (percentage or Z-Score).
                                """
                                if is_zscore:
                                    return a_m - a_b
                                elif is_pct:
                                    return (a_m - a_b) * 100
                                else:
                                    return (
                                        ((a_m - a_b) / a_b * 100)
                                        if a_b and a_b != 0
                                        else 0
                                    )

                            imp1 = calc_imp(avg_m, avg_b1)
                            imp2 = calc_imp(avg_m, avg_b2)
                            delta = round(imp2 - imp1, 2)

                            row_data.append(delta)
                            row_text.append(f"{delta:+.2f}")
                        data.append(row_data)
                        text.append(row_text)

                    df_hm = pd.DataFrame(data, columns=metrics, index=methods)
                    # Removed column normalization so absolute percentage values determine the color

                    fig = px.imshow(
                        df_hm.abs(),
                        x=metrics,
                        y=methods,
                        color_continuous_scale=["#2ca02c", "#ffbf00", "#d62728"],
                        aspect="auto",
                        title="Sensitivitäts-Check: Verzerrung der GEO-Performance durch Baseline-Schwankung (V2 vs V1)",
                    )
                    fig.update_traces(text=text, texttemplate="%{text}")
                    fig.update_coloraxes(showscale=True)
                    fig.update_xaxes(tickangle=-45)
                    return fig

                baseline_heatmap = gr.Plot(value=generate_heatmap_baseline())

            gr.Markdown("---")
            with gr.Accordion(
                "2. Überblick der Sichtbarkeitsverschiebungen der GEO-Strategien",
                open=False,
            ):
                with gr.Row():
                    metric_dropdown_overall = gr.Dropdown(
                        choices=available_metrics,
                        value=metric_init_overall,
                        label="Sichtbarkeits-Metrik auswählen",
                    )
                    focus_dropdown = gr.Dropdown(
                        choices=sorted(df_meta["Focus Topic"].unique().tolist())
                        if not df_meta.empty
                        else [],
                        multiselect=True,
                        label="Focus Topic",
                    )
                    journey_dropdown = gr.Dropdown(
                        choices=sorted(df_meta["Journey Phase"].unique().tolist())
                        if not df_meta.empty
                        else [],
                        multiselect=True,
                        label="Journey Phase",
                    )
                    persona_dropdown = gr.Dropdown(
                        choices=sorted(df_meta["Persona"].unique().tolist())
                        if not df_meta.empty
                        else [],
                        multiselect=True,
                        label="Persona",
                    )

                overall_table = gr.HTML(value=html_init_overall)
                download_btn_overall = gr.DownloadButton(
                    "📥 Tabelle als CSV herunterladen", value=path_init_overall
                )

                with gr.Row():
                    overall_plot = gr.Plot(value=init_fig)

            gr.Markdown("---")
            with gr.Accordion(
                "3. Analyse der Leistungsunterschiede der GEO-Strategien", open=False
            ):
                gr.Markdown("--- \n # Signifikanz-Cluster (Tukey Post-Hoc Test)")
                gr.Markdown(
                    "Dieser Abschnitt analysiert die drei statistischen Performance-Cluster basierend auf den Tukey-Post-Hoc-Tests. Methoden innerhalb eines Clusters unterscheiden sich nicht signifikant voneinander (p > 0,05)."
                )

                tukey_html = """
                <div style="display: flex; justify-content: center; align-items: flex-end; height: 350px; padding: 20px; font-family: sans-serif; text-align: center; gap: 15px;">
                    <!-- Cluster 2 (Left) -->
                    <div style="flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end;">
                        <div style="margin-bottom: 15px;">
                            <strong>Cluster 2 (Mid-Tier)</strong><br>
                            <span style="font-size: 0.9em; color: #555;">LLMs.txt<br>Simple Language<br>Inverted Pyramid</span>
                        </div>
                        <div style="background: linear-gradient(180deg, #b0bec5 0%, #90a4ae 100%); width: 100%; height: 140px; border-radius: 8px 8px 0 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: flex; align-items: center; justify-content: center; color: white; font-size: 2em; font-weight: bold;">2</div>
                    </div>

                    <!-- Cluster 1 (Center) -->
                    <div style="flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end;">
                        <div style="margin-bottom: 15px;">
                            <strong>Cluster 1 (Top-Tier)</strong><br>
                            <span style="font-size: 0.9em; color: #555;">Technical Terms<br>AutoGEO<br>Statistics Optimization<br>Citing Credible Sources</span>
                        </div>
                        <div style="background: linear-gradient(180deg, #ffd54f 0%, #ffca28 100%); width: 100%; height: 220px; border-radius: 8px 8px 0 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: flex; align-items: center; justify-content: center; color: white; font-size: 2.5em; font-weight: bold;">1</div>
                    </div>

                    <!-- Cluster 3 (Right) -->
                    <div style="flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end;">
                        <div style="margin-bottom: 15px;">
                            <strong>Cluster 3 (Low-Tier)</strong><br>
                            <span style="font-size: 0.9em; color: #555;">Fluent Optimization<br>More Quotes<br>Authoritative</span>
                        </div>
                        <div style="background: linear-gradient(180deg, #d7ccc8 0%, #bcaaa4 100%); width: 100%; height: 80px; border-radius: 8px 8px 0 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5em; font-weight: bold;">3</div>
                    </div>
                </div>
                """
                gr.HTML(value=tukey_html)

                gr.Markdown("### Ausgewählte Signifikanzen (p-Werte) vs. Top-Performer")
                tukey_data = pd.DataFrame(
                    [
                        {
                            "Vergleich": "Technical Terms vs. AutoGEO",
                            "Differenz": 0.003,
                            "p-Wert (Signifikanz)": "0.986",
                            "Ergebnis": "Nicht signifikant (Cluster 1)",
                        },
                        {
                            "Vergleich": "Technical Terms vs. Statistics Opt.",
                            "Differenz": 0.041,
                            "p-Wert (Signifikanz)": "1.000",
                            "Ergebnis": "Nicht signifikant (Cluster 1)",
                        },
                        {
                            "Vergleich": "Technical Terms vs. LLMs.txt",
                            "Differenz": 0.257,
                            "p-Wert (Signifikanz)": "0.009",
                            "Ergebnis": "Signifikant (Grenze zu Cluster 2)",
                        },
                        {
                            "Vergleich": "AutoGEO vs. LLMs.txt",
                            "Differenz": 0.254,
                            "p-Wert (Signifikanz)": "0.235",
                            "Ergebnis": "Nicht signifikant (Überlappung)",
                        },
                        {
                            "Vergleich": "AutoGEO vs. More Quotes",
                            "Differenz": 0.413,
                            "p-Wert (Signifikanz)": "< 0.001",
                            "Ergebnis": "Hochsignifikant (Grenze zu Cluster 3)",
                        },
                    ]
                )
                gr.Dataframe(value=tukey_data)

            gr.Markdown("---")
            with gr.Accordion(
                "4. Veranschaulichung der Textveränderungen pro Strategie", open=False
            ):
                gr.Markdown("--- \n #             ")
                gr.Markdown(
                    "Dieser Abschnitt veranschaulicht beispielhafte Textanpassungen."
                )

                metric_init_examples = (
                    "Meta-Z-Score"
                    if "Meta-Z-Score" in metrics_tag_wise
                    else (metrics_tag_wise[0] if metrics_tag_wise else None)
                )
                metric_dropdown_examples = gr.Dropdown(
                    choices=metrics_tag_wise,
                    value=metric_init_examples,
                    label="Metrik für die Auswertung",
                )

                with gr.Accordion(
                    "Repräsentative Beispiele (Nahe am Durchschnitt)", open=False
                ):
                    examples_html_component = gr.HTML(
                        value=generate_examples_html(metric_init_examples, df_merged)
                    )

                with gr.Accordion("Top-Performer (Größte Verbesserung)", open=False):
                    top_perf_component = gr.HTML(
                        value=generate_top_performer_html(
                            metric_init_examples, df_merged
                        )
                    )

                with gr.Accordion(
                    "Worst-Performer (Größte Verschlechterung)", open=False
                ):
                    worst_perf_component = gr.HTML(
                        value=generate_worst_performer_html(
                            metric_init_examples, df_merged
                        )
                    )

                metric_dropdown_examples.change(
                    fn=lambda m: (
                        generate_examples_html(m, df_merged),
                        generate_top_performer_html(m, df_merged),
                        generate_worst_performer_html(m, df_merged),
                    ),
                    inputs=[metric_dropdown_examples],
                    outputs=[
                        examples_html_component,
                        top_perf_component,
                        worst_perf_component,
                    ],
                )

            gr.Markdown("---")
            with gr.Accordion(
                "5. Einfluss von Fokusthema, Persona und Customer-Journey-Phase",
                open=False,
            ):
                gr.Markdown("### Detaillierte Aufschlüsselung nach Segmenten")
                with gr.Accordion("Auswertung nach Focus Topic", open=False):
                    topic_table = gr.HTML(value=init_topic)
                    topic_plot = gr.Plot(value=init_topic_fig)
                with gr.Accordion("Auswertung nach Persona", open=False):
                    persona_table = gr.HTML(value=init_persona)
                    persona_plot = gr.Plot(value=init_persona_fig)
                with gr.Accordion("Auswertung nach Journey Phase", open=False):
                    journey_table = gr.HTML(value=init_journey)
                    journey_plot = gr.Plot(value=init_journey_fig)

            gr.Markdown("---")
            with gr.Accordion(
                "6. Analyse der einzelnen Sichtbarkeitsmetriken", open=False
            ):
                gr.Markdown("--- \n #             ")
                gr.Markdown(
                    "Dieser Abschnitt präsentiert die absoluten, un-standardisierten Rohwerte für alle Metriken (Mean ± Std)."
                )

                def generate_absolute_table():
                    """Generate a summary table of absolute values across various evaluation modes.

                    Computes aggregated scores for all methods relative to both the
                    Average Baseline and individual Tag properties.

                    Returns:
                        pd.DataFrame: Formatted DataFrame containing absolute scores and statistics.
                    """
                    if df_merged.empty:
                        return "<p>Keine Daten.</p>"

                    base_metrics = [
                        m
                        for m in df_merged["Metric"].unique()
                        if m not in ["Meta-Z-Score", "ROUGE-L Recall", "BLEU Score"]
                    ]
                    metrics = base_metrics + [
                        m
                        for m in ["ROUGE-L Recall", "BLEU Score"]
                        if m in df_merged["Metric"].unique()
                    ]

                    meta_leaderboard_df = _cached_calculate_leaderboard(
                        "Meta-Z-Score", (), (), ()
                    )
                    ranked_methods = meta_leaderboard_df["GEO-Methode"].tolist()
                    methods = [
                        m
                        for m in ranked_methods
                        if m not in ["Baseline", "Baseline V2", "Average Baseline"]
                    ] + ["Average Baseline"]

                    units = {
                        "Chat Cite": "%",
                        "Sentence Cite": "%",
                        "Absolute Wordcount": "Wörter",
                        "Position Adjusted Wordcount": "Score",
                        "Position Adjusted Sentence Count": "Score",
                        "Length Ratio": "Faktor",
                        "Perplexity": "Score",
                        "ROUGE-L Recall": "%",
                        "BLEU Score": "%",
                        "RVS Sentiment": "Score",
                        "GEval Overall": "Score",
                    }

                    baseline_df = df_merged[df_merged["Method"] == "Average Baseline"]
                    baseline_means = {}
                    for metric in metrics:
                        vals = baseline_df[baseline_df["Metric"] == metric][
                            "Target_Score"
                        ].dropna()
                        baseline_means[metric] = vals.mean() if not vals.empty else 0

                    results = []
                    for method in methods:
                        row = {"GEO-Methode": method}
                        method_df = df_merged[df_merged["Method"] == method]
                        for metric in metrics:
                            vals = method_df[method_df["Metric"] == metric][
                                "Target_Score"
                            ].dropna()
                            if not vals.empty:
                                mean = vals.mean()
                                row[metric] = f"{mean:.2f}"
                            else:
                                row[metric] = "-"
                        results.append(row)
                    df = pd.DataFrame(results)

                    rename_dict = {}
                    for metric in metrics:
                        unit = units.get(metric, "")
                        bl_val = baseline_means.get(metric, 0)
                        rename_dict[metric] = (
                            f"{metric} [{unit}]<br>(Baseline: {bl_val:.2f})"
                        )
                    df.rename(columns=rename_dict, inplace=True)
                    cols_order = ["GEO-Methode"] + [
                        rename_dict[m] for m in metrics if m in rename_dict
                    ]
                    df = df[cols_order]
                    html = df.to_html(
                        index=False,
                        border=1,
                        classes="table table-striped",
                        justify="center",
                        escape=False,
                    )
                    return f'<div style="overflow-x: auto; white-space: nowrap;">{html}</div>'

                gr.HTML(value=generate_absolute_table())

                gr.Markdown(
                    "Diese Auswertung berechnet für jede Sichtbarkeitsmetrik den Z-Score (Effektstärke vs. Baseline). Anschließend wird der Durchschnitt dieser Z-Scores gebildet (Meta-Z-Score)."
                )

                def generate_meta_zscore_leaderboard():
                    """Generate the Meta-Z-Score overview leaderboard.

                    Compiles the baseline-adjusted Meta-Z-Score metrics to create the final
                    risk-adjusted evaluation table across all methods.

                    Returns:
                        pd.DataFrame: Meta-Z-Score leaderboard formatted for display.
                    """
                    if df_merged.empty:
                        return "<p>Keine Daten.</p>", None, None

                    all_metrics = [
                        m for m in df_merged["Metric"].unique() if m != "Meta-Z-Score"
                    ]
                    key_metrics = [
                        m
                        for m in all_metrics
                        if m not in ["Length Ratio", "Perplexity"]
                    ]
                    methods = [
                        m
                        for m in df_merged["Method"].unique()
                        if m not in ["Baseline", "Baseline V2", "Average Baseline"]
                    ]
                    baseline_df = df_merged[df_merged["Method"] == "Average Baseline"]

                    results = []
                    box_data = []
                    meta_leaderboard_df = _cached_calculate_leaderboard(
                        "Meta-Z-Score", (), (), ()
                    )
                    meta_dict = dict(
                        zip(
                            meta_leaderboard_df["GEO-Methode"],
                            meta_leaderboard_df["Effektstärke (Z-Score)"],
                        )
                    )

                    for method in methods:
                        method_df = df_merged[df_merged["Method"] == method]
                        row = {"GEO-Methode": method}

                        z_scores = []
                        for metric in all_metrics:
                            b_vals = baseline_df[baseline_df["Metric"] == metric][
                                "Target_Score"
                            ].dropna()
                            b_mean = b_vals.mean()
                            b_std = b_vals.std()

                            m_vals = method_df[method_df["Metric"] == metric][
                                "Target_Score"
                            ].dropna()
                            m_mean = m_vals.mean()

                            z = 0
                            if (
                                pd.notnull(b_std)
                                and b_std > 0
                                and pd.notnull(b_mean)
                                and pd.notnull(m_mean)
                            ):
                                z = (m_mean - b_mean) / b_std

                            row[metric] = round(z, 3)

                            if metric in key_metrics:
                                z_scores.append(z)
                                box_data.append(
                                    {
                                        "GEO-Methode": method,
                                        "Metric": metric,
                                        "Z-Score": z,
                                    }
                                )

                        meta_z = meta_dict.get(
                            method, sum(z_scores) / len(z_scores) if z_scores else 0
                        )
                        meta_std = np.std(z_scores) if z_scores else 0
                        row["Meta-Z-Score (Avg)"] = round(meta_z, 4)
                        row["Z-Score (StdDev)"] = round(meta_std, 4)
                        results.append(row)

                    res_df = pd.DataFrame(results)
                    if res_df.empty:
                        return "<p>Keine Daten.</p>", None, None

                    cols = ["GEO-Methode", "Meta-Z-Score (Avg)", "Z-Score (StdDev)"] + [
                        c
                        for c in res_df.columns
                        if c
                        not in ["GEO-Methode", "Meta-Z-Score (Avg)", "Z-Score (StdDev)"]
                    ]
                    res_df = res_df[cols]
                    res_df = res_df.sort_values(
                        by="Meta-Z-Score (Avg)", ascending=False
                    ).reset_index(drop=True)

                    tmp_path = "meta_zscore_leaderboard.csv"
                    res_df.to_csv(tmp_path, sep=";", index=False)
                    html_table = res_df.to_html(
                        index=False, border=1, classes="table table-striped"
                    )

                    # Use pre-calculated Meta-Z-Score from df_merged (ensures listwise deletion consistency)
                    meta_scores = df_merged[
                        df_merged["Metric"] == "Meta-Z-Score"
                    ].copy()
                    meta_scores.rename(
                        columns={"Target_Score": "Meta-Z-Score"}, inplace=True
                    )

                    box_df = meta_scores[meta_scores["Method"].isin(methods)]

                    import plotly.express as px

                    fig = px.box(
                        box_df,
                        y="Method",
                        x="Meta-Z-Score",
                        color="Method",
                        title="Verteilung des Meta-Z-Score über alle 600 Prompts",
                        category_orders={"Method": res_df["GEO-Methode"].tolist()},
                    )
                    fig.add_vline(x=0, line_dash="dash", line_color="black")

                    # Calculate descriptive stats for the table
                    methods_with_baseline = ["Average Baseline"] + methods
                    summary_results = []

                    # Pre-extract baseline scores for Delta calculation to match calculate_leaderboard EXACTLY
                    baseline_scores = meta_scores[
                        meta_scores["Method"] == "Average Baseline"
                    ][["Prompt_ID", "Meta-Z-Score"]].set_index("Prompt_ID")
                    baseline_scores.rename(
                        columns={"Meta-Z-Score": "Baseline_Score"}, inplace=True
                    )

                    for method in methods_with_baseline:
                        method_df = meta_scores[meta_scores["Method"] == method].copy()
                        vals = method_df["Meta-Z-Score"].dropna()

                        if not vals.empty:
                            # Calculate Delta vs Baseline per prompt
                            comp = pd.merge(
                                method_df, baseline_scores, on="Prompt_ID", how="inner"
                            )
                            comp["Delta"] = (
                                comp["Meta-Z-Score"] - comp["Baseline_Score"]
                            )

                            wcp = (comp["Delta"] > 0).mean() * 100
                            dr = (comp["Delta"] < 0).mean() * 100
                            wtr = (wcp / dr) if dr > 0 else (wcp if wcp > 0 else 0)

                            summary_results.append(
                                {
                                    "GEO-Methode": method,
                                    "Mean": vals.mean(),
                                    "StdDev": vals.std(),
                                    "Min": vals.min(),
                                    "25%": vals.quantile(0.25),
                                    "Median": vals.median(),
                                    "75%": vals.quantile(0.75),
                                    "Max": vals.max(),
                                    "Verbesserungsrate (WCP %)": wcp,
                                    "Verschlechterungsrate (DR %)": dr,
                                    "Nutzen-Risiko-Verhältnis (WTR)": wtr,
                                }
                            )
                    df_sum = pd.DataFrame(summary_results)

                    wcp_fig = None
                    if not df_sum.empty:
                        # Create Tornado chart for WCP vs DR
                        import plotly.graph_objects as go

                        fig_df = df_sum[
                            df_sum["GEO-Methode"] != "Average Baseline"
                        ].copy()

                        wcp_fig = go.Figure()
                        wcp_fig.add_trace(
                            go.Bar(
                                y=fig_df["GEO-Methode"],
                                x=fig_df["Verbesserungsrate (WCP %)"],
                                name="Verbesserungsrate (WCP %)",
                                orientation="h",
                                marker=dict(color="#28a745"),
                                text=fig_df["Verbesserungsrate (WCP %)"].apply(
                                    lambda x: f"{x:.1f}%"
                                ),
                                textposition="inside",
                            )
                        )
                        wcp_fig.add_trace(
                            go.Bar(
                                y=fig_df["GEO-Methode"],
                                x=-fig_df["Verschlechterungsrate (DR %)"],
                                name="Verschlechterungsrate (DR %)",
                                orientation="h",
                                marker=dict(color="#dc3545"),
                                text=fig_df["Verschlechterungsrate (DR %)"].apply(
                                    lambda x: f"{x:.1f}%"
                                ),
                                textposition="inside",
                            )
                        )

                        # Match the sort order of the boxplot (best method at the top)
                        ordered_methods = res_df["GEO-Methode"].tolist()
                        ordered_methods.reverse()  # Reverse so the best is at the top of the y-axis

                        wcp_fig.update_layout(
                            title="Verbesserungsrate (WCP %) vs. Verschlechterungsrate (DR %) inkl. Nutzen-Risiko-Verhältnis (WTR)",
                            barmode="relative",
                            yaxis_title="",
                            xaxis_title="Prozent (%)",
                            xaxis=dict(
                                tickvals=[-100, -50, 0, 50, 100],
                                ticktext=["100%", "50%", "0%", "50%", "100%"],
                            ),
                            yaxis=dict(
                                categoryorder="array", categoryarray=ordered_methods
                            ),
                            margin=dict(r=100),  # Margin for short annotations
                        )
                        # Add WTR annotations
                        for i, row in fig_df.iterrows():
                            wcp_fig.add_annotation(
                                x=max(fig_df["Verbesserungsrate (WCP %)"]) + 5,
                                y=row["GEO-Methode"],
                                text=f"WTR: {row['Nutzen-Risiko-Verhältnis (WTR)']:.2f}",
                                showarrow=False,
                                font=dict(color="black", size=12),
                                xanchor="left",
                            )

                    if not df_sum.empty:
                        for col in df_sum.columns:
                            if col != "GEO-Methode":
                                df_sum[col] = df_sum[col].apply(lambda x: f"{x:.4f}")
                        summary_html = df_sum.to_html(
                            index=False,
                            border=1,
                            classes="table table-striped",
                            justify="center",
                        )
                        summary_html = f'<div style="overflow-x: auto; white-space: nowrap;">{summary_html}</div>'
                    else:
                        summary_html = "<p>Keine Daten.</p>"

                    return (
                        f'<div style="overflow-x: auto; white-space: nowrap;">{html_table}</div>',
                        tmp_path,
                        fig,
                        summary_html,
                        wcp_fig,
                    )

                meta_html, meta_path, meta_fig, meta_summary_html, meta_wcp_fig = (
                    generate_meta_zscore_leaderboard()
                )
                gr.HTML(value=meta_html)
                gr.DownloadButton(
                    "📥 Meta-Z-Score Tabelle herunterladen", value=meta_path
                )
                if meta_fig:
                    gr.Plot(value=meta_fig)
                if meta_wcp_fig:
                    gr.Markdown("### Gewinn- vs. Verlustquote (Tornado-Chart)")
                    gr.Markdown(
                        "Dieser Abschnitt visualisiert die Siegquote (WCP, grün, rechts) gegen die Schadensquote (DR, rot, links). Die WTR (Win-to-Degradation Ratio) am rechten Rand zeigt das Verhältnis: Wie viele Gewinne einem Verlust gegenüberstehen."
                    )
                    gr.Plot(value=meta_wcp_fig)
                gr.Markdown(
                    "Dieser Abschnitt liefert Zusammenfassungs-Statistiken (Mean, Min, Max, Standardabweichung) für Meta-Z-Score über alle Optimierungsmethoden hinweg. So lässt sich die Vorhersehbarkeit und Varianz der Methoden vergleichen."
                )
                gr.HTML(value=meta_summary_html)

                gr.Markdown("--- \n #             ")
                gr.Markdown(
                    "Dieser Abschnitt liefert Zusammenfassungs-Statistiken (Mean, Min, Max, Standardabweichung) für eine ausgewählte Metrik über alle Optimierungsmethoden hinweg. So lässt sich die Vorhersehbarkeit und Varianz der Methoden vergleichen."
                )

                def get_summary_statistics(metric):
                    """Calculate descriptive statistics for a specific metric across methods.

                    Args:
                        metric (str): The metric to summarize.

                    Returns:
                        tuple: HTML table of descriptive statistics and a Boxplot figure.
                    """
                    if df_merged.empty or not metric:
                        return "<p>Keine Daten verfügbar.</p>"

                    metric_df = df_merged[df_merged["Metric"] == metric]
                    methods = ["Average Baseline"] + [
                        m
                        for m in df_merged["Method"].unique()
                        if m not in ["Baseline", "Baseline V2", "Average Baseline"]
                    ]

                    results = []
                    for method in methods:
                        vals = metric_df[metric_df["Method"] == method][
                            "Target_Score"
                        ].dropna()
                        if not vals.empty:
                            results.append(
                                {
                                    "GEO-Methode": method,
                                    "Mean": vals.mean(),
                                    "StdDev": vals.std(),
                                    "Min": vals.min(),
                                    "25%": vals.quantile(0.25),
                                    "Median": vals.median(),
                                    "75%": vals.quantile(0.75),
                                    "Max": vals.max(),
                                }
                            )
                    df_res = pd.DataFrame(results)
                    if not df_res.empty:
                        for col in df_res.columns:
                            if col != "GEO-Methode":
                                df_res[col] = df_res[col].apply(lambda x: f"{x:.4f}")
                        html = df_res.to_html(
                            index=False,
                            border=1,
                            classes="table table-striped",
                            justify="center",
                        )
                        return f'<div style="overflow-x: auto; white-space: nowrap;">{html}</div>'
                    return "<p>Keine Daten.</p>"

                metric_init_stat = available_metrics[0] if available_metrics else None
                stat_metric_dropdown = gr.Dropdown(
                    choices=available_metrics, value=metric_init_stat, label="Metrik"
                )
                stat_html = gr.HTML(value=get_summary_statistics(metric_init_stat))
                stat_metric_dropdown.change(
                    fn=get_summary_statistics,
                    inputs=[stat_metric_dropdown],
                    outputs=[stat_html],
                )

                gr.Markdown("--- \n #             ")
                gr.Markdown(
                    "Dieser Abschnitt analysiert die **prozentuale Steigerung** der reinen Sichtbarkeits-Metriken im Vergleich zur Average Baseline."
                )

                def generate_percentage_table():
                    """Generate a summary table displaying percentage or standard score changes.

                    Returns:
                        str: HTML representation of the percentage changes table.
                    """
                    if df_merged.empty:
                        return "<p>Keine Daten.</p>"

                    exclude_metrics = [
                        "Meta-Z-Score",
                        "Length Ratio",
                        "Perplexity",
                        "ROUGE-L Recall",
                        "BLEU Score",
                    ]
                    base_metrics = [
                        m
                        for m in df_merged["Metric"].unique()
                        if m not in exclude_metrics
                    ]
                    metrics = base_metrics + [
                        m
                        for m in ["ROUGE-L Recall", "BLEU Score"]
                        if m in df_merged["Metric"].unique()
                    ]

                    methods = [
                        m
                        for m in df_merged["Method"].unique()
                        if m not in ["Baseline", "Baseline V2", "Average Baseline"]
                    ]

                    baseline_means = {}
                    for metric in metrics:
                        b_vals = df_merged[
                            (df_merged["Method"] == "Average Baseline")
                            & (df_merged["Metric"] == metric)
                        ]["Target_Score"].dropna()
                        baseline_means[metric] = (
                            b_vals.mean() if not b_vals.empty else None
                        )

                    results = []
                    for method in methods:
                        row = {"GEO-Methode": method}
                        method_df = df_merged[df_merged["Method"] == method]

                        pct_increases = []
                        for metric in metrics:
                            b_mean = baseline_means[metric]
                            vals = method_df[method_df["Metric"] == metric][
                                "Target_Score"
                            ].dropna()

                            if (
                                not vals.empty
                                and b_mean is not None
                                and abs(b_mean) > 0.0001
                            ):
                                m_mean = vals.mean()
                                pct = ((m_mean - b_mean) / abs(b_mean)) * 100
                                row[metric] = f"{pct:+.1f} %"
                                if metric not in ["BLEU Score", "ROUGE-L Recall"]:
                                    pct_increases.append(pct)
                            else:
                                row[metric] = "-"

                        if pct_increases:
                            overall = sum(pct_increases) / len(pct_increases)
                            row["Overall (Ø Steigerung)"] = f"**{overall:+.1f} %**"
                        else:
                            row["Overall (Ø Steigerung)"] = "-"

                        results.append(row)

                    def get_overall_val(r):
                        """Extract the overall performance value for a given method row.

                        Args:
                            r (pd.Series): Data row containing metric performance.

                        Returns:
                            float: The extracted numerical overall value, or zero if missing.
                        """
                        val = r["Overall (Ø Steigerung)"]
                        if val == "-":
                            return -9999
                        m = re.search(r"([+-]?\d+\.\d+)", val)
                        if m:
                            return float(m.group(1))
                        return -9999

                    results.sort(key=get_overall_val, reverse=True)

                    cols_order = ["GEO-Methode", "Overall (Ø Steigerung)"] + metrics
                    df = pd.DataFrame(results, columns=cols_order)

                    html = df.to_html(
                        index=False,
                        border=1,
                        classes="table table-striped",
                        justify="center",
                        escape=False,
                    )
                    html = html.replace(
                        "<th>Overall", '<th style="background-color: #f0f0f0;">Overall'
                    )
                    html = html.replace(
                        "<td>**",
                        '<td style="background-color: #f8f9fa; font-weight: bold;">',
                    )
                    html = html.replace("**</td>", "</td>")

                    html += '<div style="margin-top: 10px; font-size: 0.9em; color: gray;">* Hinweis: BLEU Score und ROUGE-L Recall werden zur Berechnung der Ø Steigerung ausgeschlossen, da ihre Baselines extrem nahe an 0 liegen und dies zu mathematischen Verzerrungen (extremen Prozentwerten) führen würde.</div>'

                    return f'<div style="overflow-x: auto; white-space: nowrap;">{html}</div>'

                gr.HTML(value=generate_percentage_table())

                gr.Markdown("--- \n # Detaillierte GEval Sub-Metriken")
                gr.Markdown(
                    "Dieser Abschnitt bewertet die 7 subjektiven Qualitäts-Dimensionen des LLM-as-a-Judge (GEval)."
                )

                def generate_geval_table():
                    """Generate a detailed evaluation table for GEval metrics.

                    Computes the effect size across seven evaluation subcategories
                    and visualizes them in an HTML-formatted table.

                    Returns:
                        str: HTML table displaying absolute and delta GEval metrics.
                    """
                    geval_path = r"C:\Users\finnb\Documents\emma\Masterthesis\03_Visibility Metrics and Results\geval_absolute_results.csv"
                    target_path = r"C:\Users\finnb\Documents\emma\Masterthesis\03_Visibility Metrics and Results\target_indices.csv"
                    import os

                    if not os.path.exists(geval_path):
                        return "<p>GEval Datensatz nicht gefunden.</p>"
                    df_g = pd.read_csv(geval_path, sep=";")

                    if os.path.exists(target_path):
                        df_target = pd.read_csv(target_path, sep=";")
                        df_g = pd.merge(df_g, df_target, on="Prompt_ID", how="left")
                        df_g["Target_Index"] = (
                            df_g["Target_Index"].fillna(0).astype(int)
                        )
                    else:
                        df_g["Target_Index"] = 0

                    def get_g_target(row):
                        """Extract the source score corresponding to the target index.

                        Args:
                            row (pd.Series): A row from the raw GEval DataFrame.

                        Returns:
                            float: The score of the target index.
                        """
                        idx = int(row["Target_Index"])
                        col = f"Source_{idx}"
                        return row[col] if pd.notnull(row[col]) else 0

                    df_g["Target_Score"] = df_g.apply(get_g_target, axis=1)

                    METHOD_MAP = {
                        "Baseline": "Baseline",
                        "Baseline_V2": "Baseline V2",
                        "fluent_gpt": "Fluent Optimization",
                        "authoritative": "Authoritative",
                        "more_quotes": "More Quotes",
                        "citing_credible": "Citing Credible Sources",
                        "simple_language": "Simple Language",
                        "technical_terms": "Technical Terms",
                        "stats_optimization": "Statistics Optimization",
                        "llms_txt": "LLMs.txt",
                        "inverted_pyramid_mine": "Inverted Pyramid",
                        "autogeo_api_mine": "AutoGEO",
                    }
                    df_g["Method"] = (
                        df_g["Method"].map(METHOD_MAP).fillna(df_g["Method"])
                    )

                    agg = (
                        df_g.groupby(["Method", "Metric"])["Target_Score"]
                        .mean()
                        .reset_index()
                    )
                    pivot_df = agg.pivot(
                        index="Method", columns="Metric", values="Target_Score"
                    ).reset_index()

                    bl1 = pivot_df[pivot_df["Method"] == "Baseline"]
                    bl2 = pivot_df[pivot_df["Method"] == "Baseline V2"]
                    if not bl1.empty and not bl2.empty:
                        avg_bl = (
                            pd.concat([bl1, bl2]).mean(numeric_only=True).to_frame().T
                        )
                        avg_bl["Method"] = "Average Baseline"
                        pivot_df = pd.concat([pivot_df, avg_bl], ignore_index=True)

                    valid_methods = [
                        m
                        for m in METHOD_MAP.values()
                        if m not in ["Baseline", "Baseline V2"]
                    ] + ["Average Baseline"]
                    pivot_df = pivot_df[pivot_df["Method"].isin(valid_methods)]

                    pivot_df.rename(columns={"Method": "GEO-Methode"}, inplace=True)

                    submetrics = [c for c in pivot_df.columns if c != "GEO-Methode"]

                    overall_df = df_merged[df_merged["Metric"] == "GEval Overall"]
                    overall_means = (
                        overall_df.groupby("Method")["Target_Score"]
                        .mean()
                        .reset_index()
                    )
                    overall_means.rename(
                        columns={
                            "Target_Score": "GEval Overall",
                            "Method": "GEO-Methode",
                        },
                        inplace=True,
                    )

                    pivot_df = pd.merge(
                        pivot_df, overall_means, on="GEO-Methode", how="left"
                    )

                    cols = ["GEO-Methode", "GEval Overall"] + sorted(submetrics)
                    pivot_df = pivot_df[cols]

                    meta_leaderboard_df = _cached_calculate_leaderboard(
                        "Meta-Z-Score", (), (), ()
                    )
                    ranked_methods = meta_leaderboard_df["GEO-Methode"].tolist()
                    sort_order = [
                        m for m in ranked_methods if m in pivot_df["GEO-Methode"].values
                    ] + ["Average Baseline"]

                    pivot_df["GEO-Methode"] = pd.Categorical(
                        pivot_df["GEO-Methode"], categories=sort_order, ordered=True
                    )
                    pivot_df = pivot_df.sort_values("GEO-Methode")

                    html = pivot_df.to_html(
                        index=False,
                        border=1,
                        classes="table table-striped",
                        justify="center",
                        float_format="%.2f",
                    )
                    return f'<div style="overflow-x: auto; white-space: nowrap;">{html}</div>'

                gr.HTML(value=generate_geval_table())

                heatmap_path = r"C:\Users\finnb\Documents\emma\Masterthesis\Dashboard\geval_heatmap.png"
                import os

                if os.path.exists(heatmap_path):
                    gr.Markdown("<br><h3>Heatmap: Delta zur Baseline</h3>")
                    gr.Image(value=heatmap_path, show_label=False)

            gr.Markdown("---")
            with gr.Accordion(
                "7. Analyse der Sichtbarkeitsverschiebung zwischen den Quellen",
                open=False,
            ):
                gr.Markdown("--- \n #             ")

                def calculate_cannibalization(metric):
                    """Calculate the cannibalization effect (impact on competitor visibility).

                    Analyzes the variance in competitor scores before and after applying a
                    method to determine whether a GEO strategy degrades other sources.

                    Args:
                        metric (str): The evaluated metric.

                    Returns:
                        tuple: A tuple containing:
                            - pd.DataFrame: Dataframe showing competitor impacts.
                            - go.Figure: A Plotly Bar chart of the cannibalization effect.
                    """
                    if df_merged.empty:
                        return pd.DataFrame(), None
                    df_m = df_merged[df_merged["Metric"] == metric]
                    if df_m.empty:
                        return pd.DataFrame(), None

                    methods = [
                        m
                        for m in df_m["Method"].unique()
                        if m not in ["Average Baseline", "Baseline", "Baseline V2"]
                    ]
                    base_df = df_m[df_m["Method"] == "Average Baseline"][
                        [
                            "Prompt_ID",
                            "Target_Score",
                            "Other_Score",
                            "Comp_1_Score",
                            "Comp_2_Score",
                            "Comp_3_Score",
                            "Comp_4_Score",
                        ]
                    ].set_index("Prompt_ID")

                    results = []
                    for method in methods:
                        meth_df = df_m[df_m["Method"] == method][
                            [
                                "Prompt_ID",
                                "Target_Score",
                                "Other_Score",
                                "Comp_1_Score",
                                "Comp_2_Score",
                                "Comp_3_Score",
                                "Comp_4_Score",
                            ]
                        ].set_index("Prompt_ID")
                        comp = meth_df.join(
                            base_df, lsuffix="_m", rsuffix="_b", how="inner"
                        )
                        if comp.empty:
                            continue

                        target_delta = (
                            comp["Target_Score_m"] - comp["Target_Score_b"]
                        ).mean()
                        _ = (
                            comp["Other_Score_m"] - comp["Other_Score_b"]
                        ).mean()
                        c1_delta = (
                            comp["Comp_1_Score_m"] - comp["Comp_1_Score_b"]
                        ).mean()
                        c2_delta = (
                            comp["Comp_2_Score_m"] - comp["Comp_2_Score_b"]
                        ).mean()
                        c3_delta = (
                            comp["Comp_3_Score_m"] - comp["Comp_3_Score_b"]
                        ).mean()
                        c4_delta = (
                            comp["Comp_4_Score_m"] - comp["Comp_4_Score_b"]
                        ).mean()

                        results.append(
                            {
                                "GEO-Methode": method,
                                "Optimierte Ziel-Quelle (Delta)": round(
                                    target_delta, 4
                                ),
                                "Konkurrent 1 (Delta)": round(c1_delta, 4),
                                "Konkurrent 2 (Delta)": round(c2_delta, 4),
                                "Konkurrent 3 (Delta)": round(c3_delta, 4),
                                "Konkurrent 4 (Delta)": round(c4_delta, 4),
                            }
                        )

                    res_df = pd.DataFrame(results)
                    if res_df.empty:
                        return pd.DataFrame(), None

                    res_df = res_df.sort_values(
                        by="Optimierte Ziel-Quelle (Delta)", ascending=False
                    )
                    melt_df = res_df.melt(
                        id_vars="GEO-Methode",
                        value_vars=[
                            "Optimierte Ziel-Quelle (Delta)",
                            "Konkurrent 1 (Delta)",
                            "Konkurrent 2 (Delta)",
                            "Konkurrent 3 (Delta)",
                            "Konkurrent 4 (Delta)",
                        ],
                        var_name="Effekt",
                        value_name="Delta",
                    )

                    fig_bar = px.bar(
                        melt_df,
                        x="GEO-Methode",
                        y="Delta",
                        color="Effekt",
                        barmode="group",
                        color_discrete_map={
                            "Optimierte Ziel-Quelle (Delta)": "#2ca02c",
                            "Konkurrent 1 (Delta)": "#d62728",
                            "Konkurrent 2 (Delta)": "#ff7f0e",
                            "Konkurrent 3 (Delta)": "#1f77b4",
                            "Konkurrent 4 (Delta)": "#9467bd",
                        },
                        title=f"Verdrängungseffekt für Metrik: {metric}",
                    )
                    fig_bar.add_hline(y=0, line_dash="dash", line_color="black")

                    return res_df, fig_bar

                def generate_cannibal_heatmap():
                    """Generate a Heatmap illustrating the average cannibalization across metrics.

                    Returns:
                        go.Figure: A Heatmap visualizing the displacement of competitors across methods.
                    """
                    if df_merged.empty:
                        return None
                    metrics = [
                        m
                        for m in df_merged["Metric"].unique()
                        if m not in ["Meta-Z-Score", "Length Ratio", "Perplexity"]
                    ]
                    methods = [
                        m
                        for m in df_merged["Method"].unique()
                        if m not in ["Average Baseline", "Baseline", "Baseline V2"]
                    ]

                    hm_data = []
                    text_data = []

                    for method in methods:
                        row_vals = []
                        row_text = []
                        for metric in metrics:
                            sub_m = df_merged[
                                (df_merged["Metric"] == metric)
                                & (df_merged["Method"] == method)
                            ][["Prompt_ID", "Other_Score"]].set_index("Prompt_ID")
                            sub_b = df_merged[
                                (df_merged["Metric"] == metric)
                                & (df_merged["Method"] == "Average Baseline")
                            ][["Prompt_ID", "Other_Score"]].set_index("Prompt_ID")
                            comp = sub_m.join(
                                sub_b, lsuffix="_m", rsuffix="_b", how="inner"
                            )
                            if not comp.empty:
                                delta = (
                                    comp["Other_Score_m"] - comp["Other_Score_b"]
                                ).mean()
                                row_vals.append(delta)
                                row_text.append(f"{delta:+.3f}")
                            else:
                                row_vals.append(0)
                                row_text.append("0")
                        hm_data.append(row_vals)
                        text_data.append(row_text)

                    df_hm = pd.DataFrame(hm_data, columns=metrics, index=methods)
                    # Normalize columns to prevent one metric dominating the heatmap color scale
                    color_df = df_hm.apply(
                        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x,
                        axis=0,
                    )

                    fig_hm = px.imshow(
                        color_df,
                        x=metrics,
                        y=methods,
                        color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0,
                        aspect="auto",
                        title="Heatmap: Stärke der Konkurrenz-Verdrängung",
                    )
                    fig_hm.update_traces(text=text_data, texttemplate="%{text}")
                    fig_hm.update_coloraxes(showscale=False)
                    fig_hm.update_xaxes(tickangle=-45)
                    return fig_hm

                gr.Markdown("### Heatmap der Konkurrenz-Verdrängung über alle Metriken")
                cannibal_hm_plot = gr.Plot()

                gr.Markdown("### Detail-Ansicht nach Metrik")
                with gr.Row():
                    cannibal_metric_dropdown = gr.Dropdown(
                        choices=metrics_tag_wise,
                        value=metrics_tag_wise[0] if metrics_tag_wise else None,
                        label="Metrik für Detail-Ansicht",
                    )

                with gr.Row():
                    cannibal_plot = gr.Plot()
                cannibal_table = gr.Dataframe()

                demo.load(fn=generate_cannibal_heatmap, outputs=[cannibal_hm_plot])
                cannibal_metric_dropdown.change(
                    fn=calculate_cannibalization,
                    inputs=[cannibal_metric_dropdown],
                    outputs=[cannibal_table, cannibal_plot],
                )
                demo.load(
                    fn=calculate_cannibalization,
                    inputs=[cannibal_metric_dropdown],
                    outputs=[cannibal_table, cannibal_plot],
                )

            gr.Markdown("---")
            with gr.Accordion("8. Definitionen", open=False):
                gr.Markdown(
                    r"""
### 1. GEO-Strategien
Hier werden die 11 verschiedenen Optimierungsstrategien erklärt, die auf die Quellentexte angewendet wurden, um deren Sichtbarkeit in den LLM-Antworten zu erhöhen.

| Methode | Beschreibung |
| :--- | :--- |
| **Baseline** | Der originale, unoptimierte Quellentext (Referenzwert). |
| **fluent_gpt** (Sprachfluss) | Die Verbesserung der Lesbarkeit und Klarheit des Textes ohne inhaltliche Veränderungen. |
| **authoritative** (Autorität) | Die inhaltliche Anpassung des Tonfalls zu einem selbstbewussten und hochgradig kompetenten Expertenstil. |
| **more_quotes** (Direkte Zitate) | Die sinngemäße Ergänzung von Aussagen relevanter Autoritätspersonen im genauen Wortlaut zur Steigerung der Glaubwürdigkeit. |
| **citing_credible** (Quellenangabe) | Die natürliche Integration von Verweisen auf glaubwürdige Quellen zur Untermauerung von Aussagen. |
| **simple_language** (Einfache Sprache) | Die gezielte Reduktion der sprachlichen Komplexität für eine leichtere Verständlichkeit. |
| **technical_terms** (Technische Fachsprache) | Die Nutzung spezifischer Terminologie zur gezielten Erhöhung der inhaltlichen Tiefe. |
| **stats_optimization** (Statistiken) | Die Anreicherung des Inhalts durch objektive Fakten und exakte Zahlen. |
| **llms_txt** (maschinenlesbare Struktur) | Die Aufbereitung des Dokuments für eine optimierte Erfassbarkeit durch Sprachmodelle nach dem llms.txt-Standard. |
| **inverted_pyramid_mine** (Inverted Pyramid) | Eine Umstrukturierung des Textes, sodass die wichtigste Kernaussage als Abstract direkt am Anfang steht. |
| **autogeo_api_mine** (Kombinationsstrategie) | Die gleichzeitige Anwendung aller zuvor genannten neun Optimierungsansätze in einem einzigen Text vereint. Während AutoGEO ursprünglich als iteratives Verfahren zur schrittweisen Optimierung von Webinhalten konzipiert wurde, bezeichnet der Begriff in dieser Arbeit die kombinierte Anwendung aller neun untersuchten GEO-Strategien. Dadurch kann untersucht werden, ob sich durch die gleichzeitige Anwendung mehrerer Strategien zusätzliche Sichtbarkeitsgewinne erzielen lassen. |

---

### 2. Sichtbarkeits- und Qualitätsmetriken
Diese Metriken messen, wie stark und in welcher Qualität der eigene Quellentext in der finalen generierten Antwort des LLMs (z.B. GPT-4) auftaucht.

| Metrik | Beschreibung | Formel / Berechnung |
| :--- | :--- | :--- |
| **Absolute Wordcount** | Erfasst die reine Textmenge und Wortanzahl der relevanten Inhalte in den generierten Antworten. | $\sum (\text{Übereinstimmende Wörter})$ |
| **Position Adjusted Wordcount** | Erlaubt es, die reine Textmenge/Wortanzahl sowie die textliche Präsenz der relevanten Inhalte in den generierten Antworten präzise zu erfassen und dabei die exakte Positionierung der Nennungen zu gewichten, was einen direkten Aufschluss über die Aufmerksamkeit des Sprachmodells gibt. | $\sum (\text{Wort} \times \frac{1}{\text{Position im Text}})$ |
| **Sentence Cite** | Mit dieser Metrik lässt sich exakt bestimmen, ob und in welchem Ausmaß die generierten Antworten korrekte, satzweise Quellennachweise aufweisen. | $\sum (\text{Zitierte Sätze})$ |
| **Chat Cite** | Der prozentuale Anteil der originalen Quellensätze, die es in die Antwort geschafft haben, was ein zentrales Qualitätsmerkmal darstellt. | $\frac{\text{Anzahl zitierter Sätze}}{\text{Gesamtsätze der Originalquelle}} \cdot 100$ |
| **ROUGE-L Recall** | Basiert auf der längsten gemeinsamen Subsequenz und bewertet somit die strukturelle Ähnlichkeit sowie den Informationserhalt. | $\frac{\text{Länge der gemeinsamen Sequenz}}{\text{Gesamtlänge der Originalquelle}}$ |
| **BLEU-Score** | Misst die Präzision von Wort-N-Grammen, was die sprachliche Nähe zum Originaldokument quantifizierbar macht. | $BP \cdot \exp\left(\sum w_n \log p_n\right)$ |
| **Perplexität (Perplexity)** | Dient dazu, die Vorhersagbarkeit und die Informationsdichte der KI-Antworten mathematisch zu überprüfen, da Texte mit niedriger Perplexität (hoher Vorhersagbarkeit) von LLMs bevorzugt zitiert werden. | $e^{-\frac{1}{N} \sum \log p(w_i \vert w_{< i})}$ |
| **GEval Overall** | Ermöglicht mithilfe von LLMs als Evaluatoren eine qualitativ hochwertige, automatisierte Bewertung der generierten Texte anhand sieben vordefinierter Kriterien (Relevanz, Einfluss, Einzigartigkeit, etc.). | Skala von $1.0$ (Schlecht) bis $5.0$ (Perfekt) |
| **RVS Sentiment** | Bewertet die Sichtbarkeit der Marke oder des Unternehmens im Kontext seiner Reputation. | $-1.0$ (Negativ) bis $+1.0$ (Positiv) |
| **Length Ratio** | Dient dazu, das Längenverhältnis zwischen den generierten Texten und den Referenztexten zu kontrollieren, um Verzerrungen durch künstlich aufgeblähte Antworten zu vermeiden. | $\frac{\text{Wortanzahl der generierten Antwort}}{\text{Wortanzahl der Originalquelle}}$ |

---

### 3. Stabilitäts- und Kennzahlen (Metrics Leaderboard)
Um die Robustheit der Optimierung über verschiedene Suchanfragen hinweg zu bewerten, nutzt die Methodik risikobewusste Stabilitätsmetriken.

| Kennzahl | Beschreibung | Formel / Berechnung |
| :--- | :--- | :--- |
| **Meta-Z-Score** | Normiert alle Metriken und bildet den Durchschnitt. | $\frac{1}{n} \sum_{i=1}^{n} \frac{x_i - \mu_{Baseline}}{\sigma_{Baseline}}$ |
| **WCP (Win/Change Pct.) / Win-Tie Rate** | Misst den Anteil der Suchanfragen ohne negativen Effekt und dient somit als Näherungswert für eine Pareto-Optimalität ohne unbeabsichtigte Kollateralschäden. | $\frac{\text{Anzahl Prompts mit Verbesserung}}{\text{Gesamtzahl der Prompts}} \cdot 100$ |
| **DR (Degradation Rate) / Downside Risk** | Quantifiziert das reine Verlustrisiko, indem es ausschließlich negative Abweichungen bestraft, um schädliche Einbrüche von harmloser Volatilität zu trennen. | $\frac{\text{Anzahl Prompts mit Verschlechterung}}{\text{Gesamtzahl der Prompts}} \cdot 100$ |
| **Worst-Case Performance** | Definiert eine strikte Sicherheitsuntergrenze, indem sie den maximalen singulären Sichtbarkeitsverlust erfasst. | (Minimum über alle Prompts) |
| **WTR (Win-to-Degradation)** | Das Verhältnis von Nutzen zu Risiko (Chance-Risiko-Verhältnis). Wie viele Siege einem Verlust gegenüberstehen. | $\frac{\text{WCP}}{\text{DR}}$ |
                """,
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                    ],
                )

        inputs = [
            metric_dropdown_overall,
            focus_dropdown,
            journey_dropdown,
            persona_dropdown,
        ]
        for inp in inputs:
            inp.change(
                fn=update_overall,
                inputs=inputs,
                outputs=[
                    overall_table,
                    download_btn_overall,
                    topic_table,
                    persona_table,
                    journey_table,
                    overall_plot,
                    topic_plot,
                    persona_plot,
                    journey_plot,
                ],
            )

if __name__ == "__main__":
    demo.launch()
