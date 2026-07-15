#!/usr/bin/env python3
"""
Objective 1: Arity Distribution
Recovers most of Appendix 1 -- the heatmap/KDE visualization, the Poisson
GLM hypothesis test (real vs. randomized word order), and the summary
plots -- as one runnable pipeline.

Usage:
    python src/arity_analysis.py
"""
import os
import random
import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from conllu import parse_incr

from common import OUTPUT_DIR, ensure_dir, maybe_mount_drive


def analyze_language_arity(path):
    all_nodes = []
    intervener_nodes = []
    with open(path, "r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            out_degrees = {}
            for word in sentence:
                parent = word['head']
                if parent is not None and parent != 0:
                    out_degrees[parent] = out_degrees.get(parent, 0) + 1
            for word in sentence:
                all_nodes.append(out_degrees.get(word['id'], 0))
            for word in sentence:
                h_id = word['head']
                d_id = word['id']
                if h_id == 0 or h_id is None:
                    continue
                left, right = min(h_id, d_id), max(h_id, d_id)
                for i in range(left + 1, right):
                    intervener_nodes.append(out_degrees.get(i, 0))
    return all_nodes, intervener_nodes


def process_language(path):
    """Build the (arity, condition, length) observation table for the
    Poisson GLM: condition=1 for real word order, condition=0 for a
    word-order-shuffled baseline (dependency arcs kept, only linear
    position shuffled)."""
    observations = []
    with open(path, "r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            nodes = [t for t in sentence if isinstance(t['id'], int)]
            if not nodes:
                continue
            out_degrees = {}
            for t in nodes:
                h = t['head']
                if h is not None and h != 0:
                    out_degrees[h] = out_degrees.get(h, 0) + 1

            dependencies = [(t['head'], t['id']) for t in nodes if t['head'] not in (None, 0)]

            for h_id, d_id in dependencies:
                real_dist = abs(h_id - d_id)
                real_arity_sum = sum(out_degrees.get(i, 0)
                                      for i in range(min(h_id, d_id) + 1, max(h_id, d_id)))
                observations.append({'arity': real_arity_sum, 'condition': 1, 'length': real_dist})

            positions = [t['id'] for t in nodes]
            shuffled_positions = positions[:]
            random.shuffle(shuffled_positions)
            pos_map = dict(zip(positions, shuffled_positions))

            for h_id, d_id in dependencies:
                rand_h_pos, rand_d_pos = pos_map[h_id], pos_map[d_id]
                rand_dist = abs(rand_h_pos - rand_d_pos)
                low, high = min(rand_h_pos, rand_d_pos), max(rand_h_pos, rand_d_pos)
                rand_arity_sum = sum(out_degrees.get(orig_id, 0)
                                      for orig_id, new_pos in pos_map.items() if low < new_pos < high)
                observations.append({'arity': rand_arity_sum, 'condition': 0, 'length': rand_dist})

    return pd.DataFrame(observations)


def run_heatmap_and_kde(data_dir, save_dir):
    kde_records, heatmap_records = [], []
    conllu_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.conllu'))
    for file_name in conllu_files:
        lang_name = file_name.split('_')[0]
        node_data, intv_data = analyze_language_arity(os.path.join(data_dir, file_name))
        kde_records.append(pd.DataFrame({'Arity': node_data, 'Language': lang_name}))
        distribution = pd.Series(intv_data).value_counts(normalize=True).sort_index()
        distribution.name = lang_name
        heatmap_records.append(distribution)

    combined_kde_df = pd.concat(kde_records)
    plt.figure(figsize=(14, 8))
    sns.kdeplot(data=combined_kde_df, x='Arity', hue='Language', common_norm=False, bw_adjust=2)
    plt.title("Arity Distribution across Languages")
    plt.xlim(0, 8)
    plt.savefig(os.path.join(save_dir, "arity_kde.png"))
    plt.close()

    matrix_df = pd.concat(heatmap_records, axis=1).fillna(0).T
    matrix_df = matrix_df.iloc[:, :6]
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Intervener Arity Proportions")
    plt.xlabel("Arity")
    plt.ylabel("Language")
    plt.savefig(os.path.join(save_dir, "intervener_heatmap.png"))
    plt.close()


def run_hypothesis_tests(data_dir, save_dir):
    summary_rows, all_results = [], []
    conllu_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.conllu'))
    for file_name in conllu_files:
        lang_code = file_name.split('_')[0]
        df = process_language(os.path.join(data_dir, file_name))
        model = smf.glm(formula="arity ~ condition + length", data=df, family=sm.families.Poisson()).fit()

        beta_1 = model.params['condition']
        beta_2 = model.params['length']
        z_score = model.tvalues['condition']
        p_value = model.pvalues['condition']
        is_rejected = "Reject H0" if (beta_1 < 0 and (p_value / 2) < 0.05) else "Fail to Reject"

        summary_rows.append({'Language': lang_code, 'Beta_1': round(beta_1, 4),
                              'Z_Statistic': round(z_score, 4), 'Result': is_rejected})
        all_results.append({
            'language': lang_code, 'beta_0': model.params['Intercept'], 'beta_1': beta_1,
            'beta_2': beta_2, 'z_stat': z_score, 'p_value_one_tailed': p_value / 2,
            # NOTE: the original appendix computed this as
            # `os.exp(beta_1) if hasattr(os, 'exp') else 2.71828**beta_1` --
            # os.exp doesn't exist, so hasattr(os, 'exp') is always False and
            # it silently fell through to the hardcoded-e approximation.
            # math.exp(beta_1) is the direct, correct, and exact computation.
            'irr': math.exp(beta_1),
            'decision': is_rejected,
        })

    pd.DataFrame(all_results).to_csv(os.path.join(save_dir, "poisson_regression_summary.csv"), index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return all_results


def run_summary_plots(save_dir):
    df = pd.read_csv(os.path.join(save_dir, "poisson_regression_summary.csv"))

    df_sorted = df.sort_values('beta_1')
    plt.figure(figsize=(12, 8))
    sns.pointplot(data=df_sorted, x='beta_1', y='language', linestyle='none', color='darkblue')
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Optimization Effect Size (Beta 1) by Language")
    plt.xlabel("Coefficient Value (Negative = Higher Efficiency)")
    plt.ylabel("Language")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coefficient_forest_plot.png"))
    plt.close()

    df_sorted = df.sort_values('z_stat')
    plt.figure(figsize=(12, 8))
    colors = ['green' if z < -1.645 else 'grey' for z in df_sorted['z_stat']]
    sns.barplot(data=df_sorted, x='z_stat', y='language', hue='language', palette=colors, legend=False)
    plt.axvline(-1.645, color='red', linestyle='--', label='95% Significance Threshold')
    plt.title("Statistical Strength (Z-Statistic) by Language")
    plt.xlabel("Z-Statistic")
    plt.ylabel("Language")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "z_statistic_bars.png"))
    plt.close()


def main():
    data_dir = maybe_mount_drive()
    save_dir = ensure_dir(os.path.join(OUTPUT_DIR, "Arity_Analysis"))
    run_heatmap_and_kde(data_dir, save_dir)
    run_hypothesis_tests(data_dir, save_dir)
    run_summary_plots(save_dir)


if __name__ == "__main__":
    main()
