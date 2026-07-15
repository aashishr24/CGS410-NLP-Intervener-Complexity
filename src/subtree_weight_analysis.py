#!/usr/bin/env python3
"""
Objective 2: Subtree Weight Analysis
Recovers Appendix 3 -- the intervener-subtree-weight heatmap, and the
paired real-vs-randomized Poisson GLM hypothesis test (with the two
summary bar plots).

Usage:
    python src/subtree_weight_analysis.py
"""
import os
import io
import math
import random
import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from conllu import parse_incr

from common import OUTPUT_DIR, ensure_dir, maybe_mount_drive

warnings.filterwarnings('ignore')


def get_subtree_weights(sentence):
    weights = {word['id']: 1 for word in sentence if isinstance(word['id'], int)}
    heads = {}
    for word in sentence:
        if not isinstance(word['id'], int):
            continue
        head_val = word['head']
        if isinstance(head_val, int):
            heads[word['id']] = head_val
        elif head_val is None:
            heads[word['id']] = 0
        else:
            try:
                heads[word['id']] = int(head_val)
            except (ValueError, TypeError):
                heads[word['id']] = 0

    for i in sorted(weights.keys(), reverse=True):
        h = heads.get(i)
        if isinstance(h, int) and h in weights:
            weights[h] += weights[i]
    return weights


def analyze_weight_filter(path):
    intervener_weights = []
    with io.open(path, "r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            sent_weights = get_subtree_weights(sentence)
            for word in sentence:
                if not isinstance(word['id'], int) or not isinstance(word['head'], int):
                    continue
                h_id, d_id = word['head'], word['id']
                if h_id == 0 or h_id is None:
                    continue
                left, right = min(h_id, d_id), max(h_id, d_id)
                for i in range(left + 1, right):
                    if i in sent_weights:
                        intervener_weights.append(sent_weights[i])
    return intervener_weights


def extract_obs(sentence, weights, shuffle=False):
    nodes = [word['id'] for word in sentence if isinstance(word['id'], int)]
    if shuffle:
        shuffled = random.sample(nodes, len(nodes))
        pos = {node: i for i, node in enumerate(shuffled)}
    else:
        pos = {node: i for i, node in enumerate(nodes)}

    observations = []
    for word in sentence:
        if not isinstance(word['id'], int) or not isinstance(word['head'], int):
            continue
        h_id, d_id = word['head'], word['id']
        if h_id == 0 or h_id not in pos or d_id not in pos:
            continue
        p1, p2 = pos[h_id], pos[d_id]
        left, right = min(p1, p2), max(p1, p2)
        dist = right - left
        if dist > 1:
            observations.extend(
                {'dist': dist, 'weight': weights[n_id]}
                for n_id, p_val in pos.items() if left < p_val < right
            )
    return observations


def run_model(data):
    if not data:
        return None, None, None
    df = pd.DataFrame(data)
    counts = df.groupby(['dist', 'weight']).size().reset_index(name='y')
    counts['log_dist'] = np.log(counts['dist'])
    X = sm.add_constant(counts[['weight']])
    try:
        model = sm.GLM(counts['y'], X, family=sm.families.Poisson(), offset=counts['log_dist']).fit()
        return model.params['weight'], model.bse['weight'], model.tvalues['weight']
    except Exception:
        return None, None, None


def run_heatmap(data_dir, save_dir):
    heatmap_records = []
    conllu_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.conllu'))
    for file_name in conllu_files:
        lang_name = file_name.split('_')[0]
        print("Processing Weight Filter for: " + lang_name)
        intv_data = analyze_weight_filter(os.path.join(data_dir, file_name))
        if not intv_data:
            continue
        dist = pd.Series(intv_data).value_counts(normalize=True).sort_index()
        dist.name = lang_name
        heatmap_records.append(dist)

    matrix_df = pd.concat(heatmap_records, axis=1).fillna(0).T
    matrix_df = matrix_df.iloc[:, :3]
    matrix_df.columns = ["Weight 1", "Weight 2", "Weight 3"]
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Proportion of Intervener Weights by Language")
    plt.xlabel("Subtree Weight (Nodes)")
    plt.ylabel("Language")
    plt.savefig(os.path.join(save_dir, "weight_filter_heatmap.png"))
    plt.close()


def run_hypothesis_tests_and_plots(data_dir, save_dir):
    conllu_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.conllu'))
    all_results = []
    header = "{:<6} | {:<8} | {:<8} | {:<8} | {:<12}"
    print(header.format("Lang", "Beta_R", "Wald_Z", "Z_Diff", "Conclusion"))
    print("-" * 55)

    for f_name in conllu_files:
        lang = f_name.split('_')[0]
        path = os.path.join(data_dir, f_name)
        r_obs, s_obs = [], []
        with io.open(path, "r", encoding="utf-8") as f:
            for sent in parse_incr(f):
                w = get_subtree_weights(sent)
                r_obs.extend(extract_obs(sent, w, False))
                s_obs.extend(extract_obs(sent, w, True))

        r_b, r_se, r_z = run_model(r_obs)
        s_b, s_se, s_z = run_model(s_obs)
        if r_b is None or s_b is None:
            continue

        z_diff = (r_b - s_b) / math.sqrt(r_se ** 2 + s_se ** 2)
        test1 = r_z < -1.645
        test2 = z_diff < -1.645
        conc = "Efficient" if (test1 and test2) else "Insignificant"

        all_results.append({
            'Language': lang, 'Real_Beta': r_b, 'Real_Wald_Z': r_z,
            'Rand_Beta': s_b, 'Rand_Wald_Z': s_z, 'Z_Diff': z_diff, 'Conclusion': conc,
        })
        print(header.format(lang, round(r_b, 3), round(r_z, 2), round(z_diff, 2), conc))

    if not all_results:
        return
    df_final = pd.DataFrame(all_results)
    df_final.to_csv(os.path.join(save_dir, "dual_test_results.csv"), index=False)

    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    plot_data = df_final.sort_values("Real_Beta")
    sns.barplot(x="Language", y="Real_Beta", data=plot_data, hue="Language", palette="coolwarm", legend=False)
    plt.title("Comparison of Structural Weight Decay (Beta_R) across Languages")
    plt.ylabel("Real Beta Coefficient")
    plt.xlabel("Language Code")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(save_dir, "beta_r_distribution.png"), bbox_inches='tight', dpi=300)
    plt.close()

    plot_data = df_final.sort_values("Rand_Beta")
    plt.figure(figsize=(14, 7))
    sns.barplot(data=plot_data, x="Language", y="Rand_Beta", hue="Language", palette="viridis", legend=False)
    plt.title("Comparison of Randomized Structural Weight Decay (Beta_Random) across Languages", fontsize=14)
    plt.ylabel("Random Beta Coefficient", fontsize=12)
    plt.xlabel("Language Code", fontsize=12)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(save_dir, "beta_random_distribution.png"), bbox_inches='tight', dpi=300)
    plt.close()


def main():
    data_dir = maybe_mount_drive()
    save_dir = ensure_dir(os.path.join(OUTPUT_DIR, "Subtree_Weight_Analysis"))
    run_heatmap(data_dir, save_dir)
    run_hypothesis_tests_and_plots(data_dir, save_dir)


if __name__ == "__main__":
    main()
