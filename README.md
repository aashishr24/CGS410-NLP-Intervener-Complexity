# Empirical Distribution of Intervener Complexity in Natural Language Dependencies

**Research question:** what kind of nodes intervene a dependency in natural human languages?

**Hypothesis:** intervening nodes are short, low-arity, and mostly nouns and adverbs.

> **Engineering note:** this repo previously contained only the PDF report,
> with no runnable code. All code below was transcribed from the report's
> appendix into runnable scripts, and verified by actually downloading real
> Universal Dependencies treebanks and running every script end-to-end. In
> the process, a mislabeled-language bug, a dead-code `os.exp` bug, and a
> risky Google-Drive-deletion pattern were found and fixed, and every
> script was made runnable outside Google Colab. Full details in
> [`FIXES.md`](FIXES.md).

---

## Motivation

- **Structural geometry of language:** a sentence is an array of nodes connected by dependency arcs, where a head node governs the role of a dependent node. These pairs are often not adjacent -- the words in between are the *interveners*.
- **Cognitive efficiency:** human languages are thought to reduce memory load by keeping interveners structurally simple, since holding a head node in working memory while parsing is cognitively taxing.
- **Inspiration from Dependency Length Minimization (DLM):** parallel to DLM, which shows languages prefer shorter dependencies, this project asks whether languages also optimize the *internal complexity* of the gap itself -- low arity, minimal subtree size.
- **Categorical bias:** are interveners drawn evenly from all parts of speech, or biased toward specific UPOS tags?
- **Universality:** to check these aren't an artifact of one language family, every hypothesis is tested across **20 languages** (10 per student on this project): English, Hindi, German, Spanish, French, Italian, Russian, Chinese, Japanese, Korean, Arabic, Turkish, Finnish, Greek, Swedish, Polish, Danish, Dutch, Portuguese, Bulgarian.

## Objectives & Outcomes

| Objective | Outcome |
|---|---|
| Quantitative POS Analysis | Interveners cluster in specific categories -- Nouns, Adpositions, Determiners -- and are not homogeneous across UPOS tags |
| Arity Distribution | Low-arity constraint confirmed: interveners tend to have zero arity |
| Structural Relational Mapping | Subtree simplicity confirmed: most interveners have subtree length 1 |
| Intervener Subtree Analysis | Interveners are mostly structurally independent of the head/dependent pair |

**Conclusion:** intervening nodes are of low arity, structurally simple (subtree length 1), and mostly independent of the head-dependent pair -- but they are mostly **Nouns and Adpositions**, not Nouns and Adverbs as originally hypothesized. The original hypothesis is therefore **partially rejected**.

## Methodology

For each objective, a Poisson GLM (or, for Objective 4's overall heterogeneity test, a likelihood-ratio test) compares **real sentences** against a **randomized baseline** -- the same dependency arcs, but with word order shuffled -- to establish that any structural simplicity observed isn't just a statistical accident of how dependency trees look in general, but a genuine property of real human word order.

No specific mathematical/parametric model of language is assumed; this is closer to a large-scale, cross-linguistic hypothesis-testing exercise than a generative model.

## Objective 1 — Arity Distribution

**Arity** = number of immediate children a node has. A Poisson GLM (`arity ~ condition + length`, condition = real vs. randomized word order) tests whether real interveners have lower arity than chance would predict.

- **H₀:** β₁ = 0 (real languages are no better than random word order)
- **H₁:** β₁ < 0 (real languages are optimized for low arity)

Result: **β₁ is negative and highly significant for all 20 languages** (e.g. |Z| in the hundreds to over a thousand) -- real interveners have substantially lower arity than the shuffled baseline.

## Objective 2 — Subtree Weight Analysis

**Subtree weight** = total constituents of a node's subtree (1 if it has no children). Two separate Poisson GLMs (real vs. random) test whether heavier interveners are suppressed more than chance alone would predict.

A language is called **"Efficient"** only if it passes both:
- **Internal validity:** the real-data Wald Z < -1.645 (heavier subtrees really are rarer as interveners)
- **Optimized selection:** Z_diff < -1.645 (the suppression in real language is stronger than in the shuffled baseline, not just an artifact of large subtrees being rare in general)

Result: interveners overwhelmingly have subtree weight 1 (no children of their own), and this suppression is stronger in real word order than in the randomized baseline, across all 20 languages.

## Objective 3 — Structural Relational Mapping

Tests whether interveners tend to modify (be a child of) the head or dependent of the pair they sit between, versus being **structurally independent** of both. A Poisson GLM contrasts Head-Modifier and Dependent-Modifier rates against the Independent baseline.

Result: **interveners are overwhelmingly structurally independent** of both the head and dependent node, across all 20 languages (all Z-statistics strongly negative, rejecting the null that they're equally likely to modify either).

## Objective 4 — Quantitative POS (UPOS) Analysis

Tests (a) whether interveners are drawn homogeneously from all UPOS categories (they aren't -- an omnibus likelihood-ratio test rejects uniformity for every language), and (b) whether **NOUN** and **ADP** specifically dominate over the next most common tag (Wald Z test, one-tailed).

Result: interveners are strongly biased toward **Nouns and Adpositions**, both in the real pooled data and (to a lesser but still substantial degree) even after shuffling sentences across languages while preserving UPOS identity -- meaning part of the bias is just these tags being common overall, but a real language-order effect remains on top of that.

## Repository Structure

```
.
├── src/
│   ├── common.py                    # shared config (local-first, optional Colab/Drive)
│   ├── download_data.py             # Appendix 0: fetch the 20 UD treebanks
│   ├── arity_analysis.py            # Objective 1
│   ├── subtree_weight_analysis.py   # Objective 2
│   ├── dependency_analysis.py       # Objective 3
│   └── upos_analysis.py             # Objective 4
├── data/                            # UD treebanks land here (gitignored, run download_data.py)
├── outputs/                         # generated plots + CSVs land here, one subfolder per objective
├── CGS410_COURSE_PROJECT.pdf        # full written report
├── FIXES.md                         # changelog of bugs found and fixed
├── requirements.txt
├── LICENSE
└── README.md                        # this file
```

## Running It

```bash
git clone https://github.com/aashishr24/CGS410-NLP-Intervener-Complexity.git
cd CGS410-NLP-Intervener-Complexity
pip install -r requirements.txt

python src/download_data.py          # downloads all 20 UD treebanks into ./data
python src/arity_analysis.py         # Objective 1
python src/subtree_weight_analysis.py # Objective 2
python src/dependency_analysis.py    # Objective 3
python src/upos_analysis.py          # Objective 4
```

Each script writes its plots and CSVs to `outputs/<Objective>/`. Everything
runs as plain local Python -- no Google Colab or Drive required. (If you
do run these inside Colab, they'll automatically offer to mount Drive and
use it instead of the local `data/`/`outputs/` folders.)

Verified by actually downloading two real treebanks (Danish, Bulgarian)
and running all four scripts end-to-end -- see [`FIXES.md`](FIXES.md) for
the real output and a discussion of what was fixed.

## Dependencies

- **conllu** — CoNLL-U treebank parsing
- **networkx** — dependency-tree graph structure (Objective 4)
- **pandas / numpy** — data wrangling
- **statsmodels / patsy** — Poisson GLMs, hypothesis testing
- **scipy** — chi-squared distribution for likelihood-ratio tests
- **matplotlib / seaborn** — visualization
- **requests** — downloading the UD treebanks

Python 3.8+. See `requirements.txt` for exact versions.

## Data Source

[Universal Dependencies](https://universaldependencies.org/) Treebanks (`.conllu` train splits), one per language, fetched directly from the UD GitHub repositories.

## License

MIT
