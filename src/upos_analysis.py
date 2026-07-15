#!/usr/bin/env python3
"""
Objective 4: Quantitative POS (UPOS) Analysis
Recovers Appendix 4 -- the UPOS-distribution heatmap, the per-language and
pooled Poisson GLM analyses (with global + shuffled-baseline IRR plots),
and the Noun/Adposition dominance hypothesis tests.

Usage:
    python src/upos_analysis.py
"""
import os
import io
import random

import networkx as nx
import conllu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

from common import OUTPUT_DIR, ensure_dir, maybe_mount_drive

# NOTE: the original appendix's lang_map was missing 'BG': 'Bulgarian'
# (one of this project's actual 20 target languages) and instead had a
# stray, unused 'RO': 'Romanian' entry. Fixed below.
LANG_MAP = {
    'EN': 'English', 'HI': 'Hindi', 'DE': 'German', 'FR': 'French', 'ES': 'Spanish',
    'IT': 'Italian', 'RU': 'Russian', 'ZH': 'Chinese', 'JA': 'Japanese', 'KO': 'Korean',
    'AR': 'Arabic', 'TR': 'Turkish', 'FI': 'Finnish', 'EL': 'Greek', 'SV': 'Swedish',
    'PL': 'Polish', 'DA': 'Danish', 'NL': 'Dutch', 'PT': 'Portuguese', 'BG': 'Bulgarian',
}


class ComputeMeasuresRand:
    """Finds the intervening node ids that sit between a (head, dependent) edge."""

    def __init__(self, tree, root):
        self.tree = tree
        self.root = root

    def get_interveners(self, edge):
        nodes = sorted(n for n in self.tree.nodes if n != self.root)
        h, d = edge
        if h > d:
            return [n for n in nodes if d < n < h]
        return [n for n in nodes if h < n < d]


def conduct_pos_analysis(directory):
    files = sorted(f for f in os.listdir(directory) if f.endswith('.conllu'))
    master_data = []
    for file in files:
        lang_code = file.split('_')[0].upper()
        file_path = os.path.join(directory, file)
        full_name = LANG_MAP.get(lang_code, lang_code)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for sent in conllu.parse_incr(f):
                    G = nx.DiGraph()
                    G.add_node(0, upos="ROOT")
                    for t in sent:
                        if isinstance(t['id'], int):
                            G.add_node(t['id'], upos=t['upostag'])
                    for t in sent:
                        if isinstance(t['id'], int):
                            G.add_edge(t['head'], t['id'])
                    measurer = ComputeMeasuresRand(G, 0)
                    for h, d in G.edges():
                        if h == 0:
                            continue
                        for int_id in measurer.get_interveners((h, d)):
                            tag = G.nodes[int_id].get('upos', 'UNKNOWN')
                            master_data.append({"Language": full_name, "UPOS": tag})
        except Exception as e:
            print(f"Failed on {file}: {e}")
    return pd.DataFrame(master_data)


def run_heatmap(data_dir, save_dir):
    df_pos = conduct_pos_analysis(data_dir)
    heatmap_data = df_pos.groupby(['Language', 'UPOS']).size().unstack(fill_value=0)
    heatmap_normalized = heatmap_data.div(heatmap_data.sum(axis=1), axis=0).round(2)

    plt.figure(figsize=(20, 12))
    sns.heatmap(heatmap_normalized, cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.5, linecolor='gray', annot_kws={"size": 8})
    plt.title("Intervener Distribution Across UPOS Tags (All Languages)", fontsize=18)
    plt.xlabel("UPOS Tags", fontsize=14)
    plt.ylabel("Languages", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(save_dir, "upos_heatmap_annotated.png"), dpi=300, bbox_inches='tight')
    plt.close()

    heatmap_data.to_csv(os.path.join(save_dir, "upos_heatmap_raw.csv"))
    heatmap_normalized.to_csv(os.path.join(save_dir, "upos_heatmap_normalized.csv"))


def _get_counts(file_path, shuffle_words=False):
    obs = []
    with io.open(file_path, 'r', encoding='utf-8') as f:
        sent = []
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    if shuffle_words:
                        original_ids = [w['id'] for w in sent]
                        new_positions = random.sample(original_ids, len(original_ids))
                        id_map = dict(zip(original_ids, new_positions))
                        for w in sent:
                            w['id'] = id_map[w['id']]
                            if w['head'] != 0 and w['head'] in id_map:
                                w['head'] = id_map[w['head']]
                    tag_map = {w['id']: w['upos'] for w in sent}
                    for w in sent:
                        h, d = w['head'], w['id']
                        if h == 0:
                            continue
                        dist = abs(h - d)
                        start, end = min(h, d), max(h, d)
                        for i in range(start + 1, end):
                            if i in tag_map:
                                obs.append({'Distance': dist, 'Tag': tag_map[i]})
                sent = []
                continue
            if line.startswith('#'):
                continue
            p = line.split('\t')
            if '-' in p[0] or '.' in p[0]:
                continue
            sent.append({'id': int(p[0]), 'upos': p[3], 'head': int(p[6])})
    df = pd.DataFrame(obs)
    if df.empty:
        return df
    return df.groupby(['Distance', 'Tag']).size().reset_index(name='Count')


def run_per_language_glm(data_dir, save_dir):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.conllu'))
    all_lang_results = []
    for f_name in files:
        lang_code = f_name.split('_')[0].upper()
        lang_name = LANG_MAP.get(lang_code, lang_code)
        try:
            data = _get_counts(os.path.join(data_dir, f_name))
            model = smf.glm("Count ~ Distance + C(Tag, Treatment(reference='VERB'))",
                             data=data, family=sm.families.Poisson()).fit()
            row = {'Language': lang_name, 'Beta_Nought': model.params['Intercept']}
            for param_name, value in model.params.items():
                if 'T.' in param_name:
                    tag = param_name.split('.')[-1][:-1]
                    row['IRR_' + tag] = np.exp(value)
            row['IRR_VERB'] = 1.0
            all_lang_results.append(row)
            print(f"Processed: {lang_name}")
        except Exception as e:
            print(f"Failed {lang_name}: {e}")

    final_df = pd.DataFrame(all_lang_results)
    cols = ['Language', 'Beta_Nought'] + [c for c in final_df.columns if c.startswith('IRR_')]
    final_df = final_df[cols].fillna(0)
    final_df.to_csv(os.path.join(save_dir, "upos_glm_summary_table.csv"), index=False)
    print("\n--- Final Summary Table (First 10 Rows) ---")
    print(final_df.head(10))


def _pooled_irr_plot(data_dir, save_dir, shuffle_words, title, out_name, csv_name):
    files = [f for f in os.listdir(data_dir) if f.endswith('.conllu')]
    print(f"Parsing {len(files)} files (Shuffling={shuffle_words})...")
    frames = []
    for filename in files:
        counts = _get_counts(os.path.join(data_dir, filename), shuffle_words=shuffle_words)
        if not counts.empty:
            frames.append(counts)
    if not frames:
        print("No data parsed.")
        return

    global_counts = pd.concat(frames).groupby(['Distance', 'Tag'], as_index=False)['Count'].sum()
    model = smf.glm("Count ~ Distance + C(Tag, Treatment(reference='VERB'))",
                     data=global_counts, family=sm.families.Poisson()).fit()

    results = pd.DataFrame({'Beta': model.params}).reset_index()
    results.columns = ['index', 'Beta']
    results['IRR'] = np.exp(results['Beta'])
    results['UPOS'] = results['index'].apply(
        lambda x: x.split('.')[-1][:-1] if 'T.' in x else ('VERB' if 'Intercept' in x else x))
    results = results[results['index'] != 'Distance']
    verb_idx = results[results['UPOS'] == 'VERB'].index
    if not verb_idx.empty:
        results.loc[verb_idx[0], 'IRR'] = 1.0
    results = results.sort_values(by='IRR', ascending=False)

    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    palette = sns.color_palette("YlOrRd_r", len(results))
    ax = sns.barplot(x='IRR', y='UPOS', data=results, hue='UPOS', palette=palette, legend=False)
    plt.axvline(x=1, color='black', linestyle='--', linewidth=2, label='Baseline (VERB)')
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.2f}", (width, p.get_y() + p.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontweight='bold')
    plt.title(title, fontsize=18)
    plt.xlabel("Incident Rate Ratio (Likelihood relative to VERB)", fontsize=14)
    plt.ylabel("UPOS Tag", fontsize=14)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, out_name), dpi=300, bbox_inches='tight')
    plt.close()
    results.to_csv(os.path.join(save_dir, csv_name), index=False)
    print(f"Beta_0 (Intercept): {model.params['Intercept']:.4f}")


def run_hypothesis_tests(data_dir, save_dir):
    final_results = []
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.conllu'))
    for filename in files:
        lang_code = filename.split('_')[0].upper()
        lang_name = LANG_MAP.get(lang_code, lang_code)
        print(f"Testing: {lang_name}")
        try:
            data = _get_counts(os.path.join(data_dir, filename))
            if data.empty:
                continue
            m0 = smf.glm("Count ~ Distance", data=data, family=sm.families.Poisson()).fit()
            m1 = smf.glm("Count ~ Distance + C(Tag, Treatment(reference='VERB'))",
                         data=data, family=sm.families.Poisson()).fit()
            lrt_stat = 2 * (m1.llf - m0.llf)
            p_lrt = chi2.sf(lrt_stat, df=len(m1.params) - len(m0.params))
            reject_lrt = "Yes" if p_lrt < 0.05 else "No"

            params = m1.params.drop(['Intercept', 'Distance'])
            others = params.drop([k for k in params.index if 'NOUN' in k or 'ADP' in k])
            if others.empty:
                continue
            next_highest_tag = others.idxmax()
            beta_next = others.max()
            se_next = m1.bse[next_highest_tag]

            lang_output = {'Language': lang_name, 'LRT_Stat': lrt_stat, 'LRT_Reject_H0': reject_lrt}
            for target in ['NOUN', 'ADP']:
                matches = [k for k in m1.params.index if target in k]
                if not matches:
                    continue
                key = matches[0]
                beta_t = m1.params[key]
                se_t = m1.bse[key]
                z_val = (beta_t - beta_next) / np.sqrt(se_t ** 2 + se_next ** 2)
                reject_dom = "Yes" if z_val > 1.645 else "No"
                lang_output[target + '_Beta'] = beta_t
                lang_output[target + '_Wald_Z'] = z_val
                lang_output[target + '_Dominant'] = reject_dom
            final_results.append(lang_output)
        except Exception as e:
            print(f"Error in {lang_name}: {e}")

    results_df = pd.DataFrame(final_results)
    results_df.to_csv(os.path.join(save_dir, "hypothesis_testing_results.csv"), index=False)
    print("\n--- Final Hypothesis Test Summary ---")
    print(results_df)


def main():
    # NOTE: the original appendix force-unmounted and deleted the contents of
    # /content/drive before remounting on every run, as a way to force a
    # clean remount in Colab. That's unnecessarily destructive -- if the
    # unmount silently failed, the delete loop could wipe real Drive files.
    # `drive.mount(..., force_remount=True)` alone is Google's own supported
    # way to force a clean remount, with none of that risk.
    data_dir = maybe_mount_drive()
    save_dir = ensure_dir(os.path.join(OUTPUT_DIR, "UPOS_Analysis"))

    run_heatmap(data_dir, save_dir)
    run_per_language_glm(data_dir, save_dir)
    _pooled_irr_plot(data_dir, save_dir, shuffle_words=False,
                      title="Global Incident Rate Ratios (IRRs) of Intervening UPOS Tags\n(Pooled Analysis of All Languages)",
                      out_name="global_upos_dominance.png", csv_name="global_upos_results.csv")
    _pooled_irr_plot(data_dir, save_dir, shuffle_words=True,
                      title="Shuffled Global Incident Rate Ratios (IRRs) of Intervening UPOS Tags",
                      out_name="shuffled_upos_dominance.png", csv_name="shuffled_upos_results.csv")
    run_hypothesis_tests(data_dir, save_dir)


if __name__ == "__main__":
    main()
