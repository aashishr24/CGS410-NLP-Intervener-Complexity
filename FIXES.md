# Fixes applied to this project

This repo previously contained only the PDF report -- no runnable code had
actually been committed. All code below was extracted from the report's
appendix, reorganized into runnable scripts, and verified by actually
downloading real Universal Dependencies treebanks (Danish, Bulgarian) and
running every script end-to-end. In the process, a few real bugs in the
original appendix code were found and fixed.

## 1. Wrong language map: Bulgarian mislabeled, Romanian phantom entry (`upos_analysis.py`)

**Before:** the UPOS analysis scripts' `lang_map` dictionary was missing
`'BG': 'Bulgarian'` -- one of the project's actual 20 target languages --
and instead had an unused `'RO': 'Romanian'` entry (Romanian isn't one of
the 20 languages this project studies at all). This didn't crash anything
(`lang_map.get(lang_code, lang_code)` just fell back to displaying the raw
code `"BG"` instead of `"Bulgarian"`), but it meant every UPOS-analysis
plot, table, and CSV silently mislabeled Bulgaria's row.

**Verified:** re-ran the UPOS analysis on real Bulgarian treebank data.
Before the fix, its row would print as `BG`. After the fix, it correctly
prints as `Bulgarian`, and the actual statistics are identical either way
(e.g. `LRT_Stat` for Bulgarian: `161232.503616`, matching the original
report's table exactly) -- confirming this was a pure labeling bug, not a
computation bug.

## 2. `os.exp` doesn't exist (`arity_analysis.py`)

**Before:**
```python
'irr': os.exp(beta_1) if hasattr(os, 'exp') else 2.71828**beta_1,
```
The `os` module has no `exp` function, so `hasattr(os, 'exp')` was always
`False`, and this silently fell through to a hardcoded approximation of
*e* every single time. It happened to produce a reasonable answer, but for
the wrong reason, and the "if" branch was permanently dead code.

**Fix:** `math.exp(beta_1)` -- direct, exact, and no dead branch.

## 3. Destructive Google Drive remount pattern (`upos_analysis.py`)

**Before:** several appendix scripts force-unmounted Google Drive and then
manually deleted everything inside `/content/drive` before remounting, as
a way to force a clean session:
```python
if os.path.ismount('/content/drive'):
    try:
        drive.flush_and_unmount()
    except:
        pass
if os.path.isdir('/content/drive') and os.listdir('/content/drive'):
    for item in os.listdir('/content/drive'):
        ...
        os.unlink(item_path)  # or shutil.rmtree(item_path)
drive.mount('/content/drive', force_remount=True)
```
If the unmount silently failed (caught by a bare `except: pass`) while
Drive was still actually mounted, this delete loop could wipe real files
in the user's Google Drive.

**Fix:** `drive.mount('/content/drive', force_remount=True)` alone is
Google's own supported way to force a clean remount -- no manual deletion
needed, and no risk.

## 4. No code was actually runnable outside Google Colab

**Before:** every script hardcoded `/content/drive/MyDrive/Course_Project`
and unconditionally called `google.colab.drive.mount(...)`, so nothing in
this repo could run anywhere except inside a Colab notebook with the
author's own Drive attached.

**Fix:** added `src/common.py`, which only mounts Drive if actually
running inside Colab (`try: from google.colab import drive`), and
otherwise uses a local `./data` / `./outputs` folder. All four analysis
scripts now run as plain local Python scripts; Colab behavior is
unchanged if you do run them there.

## 5. Repo had no runnable code committed at all

Only the PDF report existed in the repo. All code in `src/` was
transcribed from the report's appendix, reorganized into one cohesive,
importable script per objective (previously the appendix had 3-5 separate,
independently-pasted Colab-cell snippets per objective that depended on
being run in a specific order in the same session), and verified against
real data as described above.

## Verification

Ran all four scripts end-to-end against real Universal Dependencies data
(Danish + Bulgarian treebanks, downloaded fresh):

```
$ python src/arity_analysis.py
Language  Beta_1  Z_Statistic    Result
      bg -1.0556    -310.8084 Reject H0
      da -1.1129    -307.8403 Reject H0

$ python src/dependency_analysis.py
Language         Z_H    Result_H         Z_D    Result_D
      da  -94.807401 Reject Null -159.793931 Reject Null
      bg -105.734438 Reject Null -141.969907 Reject Null

$ python src/subtree_weight_analysis.py
Lang   | Beta_R   | Wald_Z   | Z_Diff   | Conclusion
bg     | -0.962   | -443.54  | -176.21  | Efficient
da     | -0.901   | -412.42  | -160.32  | Efficient

$ python src/upos_analysis.py
Processed: Bulgarian
Processed: Danish
...
```

The `dependency_analysis.py` Z-statistics above are an *exact* match to
the original report's published numbers for these two languages,
confirming the consolidation/refactor didn't change any actual
computation -- only fixed the three bugs above and made everything
runnable outside Colab.
