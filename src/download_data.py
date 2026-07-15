#!/usr/bin/env python3
"""
Appendix 0: download the Universal Dependencies treebank files used by this
project (20 languages, 10 per student) and set up the expected folder
structure.

Usage:
    python src/download_data.py
"""
import os
import requests

from common import ensure_dir, maybe_mount_drive

FILES = {
    "en_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu",
    "hi_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Hindi-HDTB/master/hi_hdtb-ud-train.conllu",
    "de_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu",
    "es_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-GSD/master/es_gsd-ud-train.conllu",
    "fr_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master/fr_gsd-ud-train.conllu",
    "it_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu",
    "ru_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-GSD/master/ru_gsd-ud-train.conllu",
    "zh_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSD/master/zh_gsd-ud-train.conllu",
    "ja_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Japanese-GSD/master/ja_gsd-ud-train.conllu",
    "ko_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Korean-GSD/master/ko_gsd-ud-train.conllu",
    "ar_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master/ar_padt-ud-train.conllu",
    "tr_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-train.conllu",
    "fi_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Finnish-TDT/master/fi_tdt-ud-train.conllu",
    "el_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Greek-GDT/master/el_gdt-ud-train.conllu",
    "sv_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-Talbanken/master/sv_talbanken-ud-train.conllu",
    "pl_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Polish-PDB/master/pl_pdb-ud-train.conllu",
    "da_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Danish-DDT/master/da_ddt-ud-train.conllu",
    "nl_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Dutch-Alpino/master/nl_alpino-ud-train.conllu",
    "pt_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-GSD/master/pt_gsd-ud-train.conllu",
    "bg_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Bulgarian-BTB/master/bg_btb-ud-train.conllu",
}

SUBFOLDERS = ["Arity_Analysis", "Dependency_Analysis", "Subtree_Weight_Analysis", "UPOS_Analysis"]


def main():
    data_dir = maybe_mount_drive()
    for sub in SUBFOLDERS:
        ensure_dir(os.path.join(data_dir, sub))

    for name, url in FILES.items():
        out_path = os.path.join(data_dir, name + ".conllu")
        if os.path.exists(out_path):
            print(f"Already have {name}, skipping.")
            continue
        print(f"Downloading {name}...")
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
        else:
            print(f"Failed: {name} ({r.status_code})")
    print("All downloads complete.")


if __name__ == "__main__":
    main()
