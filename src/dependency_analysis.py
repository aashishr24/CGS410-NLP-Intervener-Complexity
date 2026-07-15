#!/usr/bin/env python3
"""
Objective 3: Structural Relational Mapping
Recovers Appendix 2 -- proving interveners are structurally independent of
the head/dependent pair they sit between. Combines the empirical heatmap,
the Head-vs-Independent / Dependent-vs-Independent hypothesis tests, and
the forest-plot / IRR / "gatekeeper hierarchy" summary plots.

Usage:
    python src/dependency_analysis.py
"""
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy.contrasts as sx

from common import OUTPUT_DIR, ensure_dir, maybe_mount_drive

LANGUAGES = ['zh', 'tr', 'sv', 'ru', 'pt', 'pl', 'nl', 'ko', 'ja', 'it',
             'hi', 'fr', 'fi', 'es', 'en', 'el', 'de', 'da', 'bg', 'ar']


def _iter_sentences(file_path):
    """Minimal manual CoNLL-U reader: yields {id: head} maps per sentence."""
    current_sent = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.strip() == '':
                if current_sent:
                    yield current_sent
                current_sent = []
            else:
                current_sent.append(line.strip().split('\t'))
    if current_sent:
        yield current_sent


def _tree_from_sentence(sent):
    return {int(node[0]): int(node[6]) for node in sent if node[0].isdigit()}


def run_heatmap(data_dir, save_dir):
    dataset = []
    for lang in LANGUAGES:
        path = os.path.join(data_dir, f"{lang}_train.conllu")
        if not os.path.exists(path):
            continue
        for sent in _iter_sentences(path):
            tree = _tree_from_sentence(sent)
            for dep, head in tree.items():
                if head == 0:
                    continue
                left, right = min(dep, head), max(dep, head)
                if right - left <= 1:
                    continue
                for i in range(left + 1, right):
                    if i not in tree:
                        continue
                    i_head = tree[i]
                    if i_head == head:
                        role = 'Head-Modifier (H)'
                    elif i_head == dep:
                        role = 'Dependent-Modifier (D)'
                    else:
                        role = 'Independent (I)'
                    dataset.append({'Language': lang, 'Category': role})

    analysis_df = pd.DataFrame(dataset)

    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=analysis_df, x=analysis_df.index, hue='Category', fill=True, common_norm=False)
    plt.title('Distribution of Intervener Categories')
    plt.xlabel('Observation Index')
    plt.ylabel('Density')
    plt.savefig(os.path.join(save_dir, 'intervener_kernel_density.png'))
    plt.close()

    counts = analysis_df.groupby(['Language', 'Category']).size().unstack(fill_value=0)
    proportions = counts.div(counts.sum(axis=1), axis=0)
    plt.figure(figsize=(14, 10))
    sns.heatmap(proportions, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Relative Frequency of Intervener Roles Across Languages')
    plt.savefig(os.path.join(save_dir, 'intervener_role_heatmap.png'))
    plt.close()


def run_hypothesis_tests(data_dir, save_dir):
    results = []
    for lang in LANGUAGES:
        file_path = os.path.join(data_dir, f"{lang}_train.conllu")
        if not os.path.exists(file_path):
            continue
        records = []
        for sent in _iter_sentences(file_path):
            nodes = _tree_from_sentence(sent)
            for dep, head in nodes.items():
                if head == 0:
                    continue
                start, end = min(dep, head), max(dep, head)
                dist = end - start
                if dist <= 1:
                    continue
                counts = {'H': 0, 'D': 0, 'I': 0}
                for i in range(start + 1, end):
                    if i not in nodes:
                        continue
                    if nodes[i] == head:
                        counts['H'] += 1
                    elif nodes[i] == dep:
                        counts['D'] += 1
                    else:
                        counts['I'] += 1
                for cat in ['H', 'D', 'I']:
                    records.append({'count': counts[cat], 'category': cat, 'log_dist': np.log(dist)})

        if not records:
            continue
        df = pd.DataFrame(records)
        model = smf.glm(
            formula="count ~ C(category, sx.Treatment(reference='I'))",
            data=df, family=sm.families.Poisson(), offset=df['log_dist']
        ).fit()

        b_h = model.params["C(category, sx.Treatment(reference='I'))[T.H]"]
        z_h = model.tvalues["C(category, sx.Treatment(reference='I'))[T.H]"]
        p_h_raw = model.pvalues["C(category, sx.Treatment(reference='I'))[T.H]"]
        p_h = p_h_raw / 2 if z_h < 0 else 1 - (p_h_raw / 2)

        b_d = model.params["C(category, sx.Treatment(reference='I'))[T.D]"]
        z_d = model.tvalues["C(category, sx.Treatment(reference='I'))[T.D]"]
        p_d_raw = model.pvalues["C(category, sx.Treatment(reference='I'))[T.D]"]
        p_d = p_d_raw / 2 if z_d < 0 else 1 - (p_d_raw / 2)

        results.append({
            'Language': lang,
            'Beta_H_vs_I': b_h, 'Z_H': z_h, 'P_H_OneTailed': p_h,
            'Result_H': 'Reject Null' if p_h < 0.05 and b_h < 0 else 'Fail to Reject',
            'Beta_D_vs_I': b_d, 'Z_D': z_d, 'P_D_OneTailed': p_d,
            'Result_D': 'Reject Null' if p_d < 0.05 and b_d < 0 else 'Fail to Reject',
        })

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(os.path.join(save_dir, "language_hypothesis_testing.csv"), index=False)
    print(summary_df[['Language', 'Z_H', 'Result_H', 'Z_D', 'Result_D']].to_string(index=False))


def run_plots(data_dir, save_dir):
    plot_data = []
    for lang in LANGUAGES:
        file_path = os.path.join(data_dir, f"{lang}_train.conllu")
        if not os.path.exists(file_path):
            continue
        observations = []
        for sent in _iter_sentences(file_path):
            nodes = _tree_from_sentence(sent)
            for dep, head in nodes.items():
                if head == 0:
                    continue
                start, end = min(dep, head), max(dep, head)
                dist = end - start
                if dist <= 1:
                    continue
                counts = {'H': 0, 'D': 0, 'I': 0}
                for i in range(start + 1, end):
                    if i in nodes:
                        if nodes[i] == head:
                            counts['H'] += 1
                        elif nodes[i] == dep:
                            counts['D'] += 1
                        else:
                            counts['I'] += 1
                for cat in ['H', 'D', 'I']:
                    observations.append({'count': counts[cat], 'category': cat, 'log_dist': np.log(dist)})

        if not observations:
            continue
        df_lang = pd.DataFrame(observations)
        model = smf.glm(formula="count ~ category - 1", data=df_lang,
                         family=sm.families.Poisson(), offset=df_lang['log_dist']).fit()
        conf = model.conf_int()
        for cat in ['H', 'D', 'I']:
            p_name = f"category[{cat}]"
            plot_data.append({
                'Language': lang, 'Category': cat, 'Beta': model.params[p_name],
                'Lower': conf.loc[p_name, 0], 'Upper': conf.loc[p_name, 1],
                'IRR': np.exp(model.params[p_name]),
            })

    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 14))
    sns.set_theme(style="whitegrid")
    colors = {'H': '#1f77b4', 'D': '#2ca02c', 'I': '#d62728'}
    for cat in ['H', 'D', 'I']:
        sub = df_plot[df_plot['Category'] == cat]
        plt.errorbar(sub['Beta'], sub['Language'],
                     xerr=[sub['Beta'] - sub['Lower'], sub['Upper'] - sub['Beta']],
                     fmt='o', label=f'Category {cat}', color=colors[cat], alpha=0.8, capsize=3)
    plt.axvline(0, color='black', linestyle='--', alpha=0.6)
    plt.title('Forest Plot: Log-Rate Coefficients (Beta) across 20 Languages', fontsize=14)
    plt.xlabel('Log-Rate Coefficient (Beta)')
    plt.legend(title='Intervener Role')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'forest_plot_betas.png'))
    plt.close()

    plt.figure(figsize=(16, 7))
    sns.barplot(data=df_plot, x='Language', y='IRR', hue='Category', palette='muted')
    plt.axhline(1, color='red', linestyle='--', label='Neutral Baseline')
    plt.title('Incidence Rate Ratio (IRR) by Structural Role', fontsize=14)
    plt.ylabel('IRR (Exp(Beta))')
    plt.xticks(rotation=45)
    plt.legend(title='Role')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'irr_grouped_barplot.png'))
    plt.close()

    pivot_slope = df_plot.pivot(index='Language', columns='Category', values='Beta')
    plt.figure(figsize=(10, 10))
    cat_order = ['I', 'D', 'H']
    cat_labels = ['Independent', 'Dependent', 'Head']
    for lang in pivot_slope.index:
        y_vals = [pivot_slope.loc[lang, c] for c in cat_order]
        plt.plot(cat_labels, y_vals, marker='o', alpha=0.4, linewidth=1.5)
        plt.text(2.05, y_vals[2], lang, fontsize=9, alpha=0.8)
    plt.title('The Gatekeeper Hierarchy: Universal Suppression Ranking', fontsize=14)
    plt.ylabel('Beta (Log-Rate of Intervention)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'structural_ranking_slopegraph.png'))
    plt.close()


def main():
    data_dir = maybe_mount_drive()
    save_dir = ensure_dir(os.path.join(OUTPUT_DIR, "Dependency_Analysis"))
    run_heatmap(data_dir, save_dir)
    run_hypothesis_tests(data_dir, save_dir)
    run_plots(data_dir, save_dir)


if __name__ == "__main__":
    main()
