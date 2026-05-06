#!/usr/bin/env python3
"""
Extraction des RCP ANSM depuis CIS_RCP.html vers un fichier Excel.

Usage :
    python3 parse_cis_rcp.py               # traitement complet
    python3 parse_cis_rcp.py --sample 100  # test sur 100 médicaments
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# Les RCP HTML peuvent dépasser la limite par défaut de 131 Ko
csv.field_size_limit(sys.maxsize)

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

OUTPUT_XLSX = Path("CIS_RCP_export.xlsx")
OUTPUT_CSV = Path("CIS_RCP_export.csv")
CHUNKSIZE = 300

def _find_input_file() -> Path:
    for name in ("CIS_RCP.html", "CIS_RCP.htm"):
        p = Path(name)
        if p.exists():
            return p
    return Path("CIS_RCP.html")  # valeur par défaut pour le message d'erreur

INPUT_FILE = _find_input_file()

# Le fichier contient deux générations de nommage d'ancres.
# Chaque entrée liste les variantes par ordre de priorité (nouvelle puis ancienne).
# Format : "colonne": ["ancre_variante_1", "ancre_variante_2", ...]
SECTIONS: dict[str, list[str]] = {
    "denomination":            ["RcpDenomination"],
    # "QualiQuanti" (new ~11500) vs "QualitQuanti" (old ~530)
    "composition":             ["RcpCompoQualiQuanti",   "RcpCompoQualitQuanti"],
    "forme_pharmaceutique":    ["RcpFormePharm"],
    "indications":             ["RcpIndicTherap"],
    "posologie":               ["RcpPosoAdmin"],
    # "Contreindications" (new) vs "ContreIndic" (old)
    "contre_indications":      ["RcpContreindications",  "RcpContreIndic"],
    "mises_en_garde":          ["RcpMisesEnGarde"],
    # "InteractionsMed" (new) vs "Interactions" (old)
    "interactions":            ["RcpInteractionsMed",    "RcpInteractions"],
    # "FertGrossAllait" (new, inclut la fertilité) vs "GrossAllait" (old)
    "grossesse_allaitement":   ["RcpFertGrossAllait",    "RcpGrossAllait"],
    "effets_indesirables":     ["RcpEffetsIndesirables"],
    "surdosage":               ["RcpSurdosage"],
    "excipients":              ["RcpListeExcipients"],
    "duree_conservation":      ["RcpDureeConservation"],
    "titulaire_amm":           ["RcpTitulaireAmm"],
    # "NumAutor" (new) vs "Presentation" (old)
    "numero_amm":              ["RcpNumAutor",           "RcpPresentation"],
    "conditions_prescription": ["RcpCondPrescription"],
}

# On s'arrête à la prochaine rubrique de même niveau ou supérieur.
# Titre3 et Titre4 sont des sous-rubriques internes, on ne s'y arrête pas.
STOP_CLASSES = {"AmmAnnexeTitre1", "AmmAnnexeTitre2"}

# Éléments de navigation/mise en page à exclure du contenu extrait
SKIP_CLASSES = {"alignright", "AmmAnnexeTitre"}

# Textes de navigation récurrents à supprimer en post-traitement
_NAV_RE = re.compile(r'\s*\|\s*Retour en haut(?: de la page)?\s*$', re.IGNORECASE)

# Limite de caractères par cellule Excel (32 767 max)
EXCEL_CELL_LIMIT = 32_000


def _extract_one_anchor(soup: BeautifulSoup, anchor_name: str) -> str:
    """Tente d'extraire le contenu textuel après une ancre nommée."""
    anchor = soup.find("a", {"name": anchor_name})
    if not anchor:
        return ""
    parent_p = anchor.find_parent("p")
    if not parent_p:
        return ""

    # Cas particulier : certains titres (ex. CONDITIONS DE PRESCRIPTION) sont dans un
    # <div> séparateur qui n'a qu'un seul enfant. Le contenu suit le <div>, pas le <p>.
    # On ne remonte PAS si le <div> est le conteneur principal (plusieurs enfants).
    start_node = parent_p
    parent_div = parent_p.parent
    if parent_div and parent_div.name == "div":
        div_element_children = [c for c in parent_div.children if getattr(c, "name", None)]
        if len(div_element_children) == 1:
            start_node = parent_div

    texts = []
    for sib in start_node.find_next_siblings():
        sib_classes = set(sib.get("class", []))
        if sib_classes & STOP_CLASSES:
            break
        if sib_classes & SKIP_CLASSES:
            continue
        text = sib.get_text(separator=" ", strip=True)
        if text:
            texts.append(text)
    return " | ".join(texts)


def extract_section(soup: BeautifulSoup, anchor_variants: list[str]) -> str:
    """Essaie chaque variante d'ancre dans l'ordre et retourne le premier résultat non vide."""
    for anchor_name in anchor_variants:
        result = _extract_one_anchor(soup, anchor_name)
        if result:
            return _NAV_RE.sub("", result).strip()
    return ""


def parse_rcp(code_cis: str, html: str) -> dict:
    """Parse le HTML d'un RCP et retourne un dictionnaire de champs extraits."""
    soup = BeautifulSoup(html, "lxml")

    date_p = soup.find("p", class_="DateNotif")
    record = {
        "code_cis": code_cis,
        "date_mise_a_jour": date_p.get_text(strip=True) if date_p else "",
    }

    for col_name, anchor_variants in SECTIONS.items():
        text = extract_section(soup, anchor_variants)
        record[col_name] = text[:EXCEL_CELL_LIMIT] if len(text) > EXCEL_CELL_LIMIT else text

    return record


def main():
    parser = argparse.ArgumentParser(description="Extraction RCP ANSM → Excel")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Ne traiter que les N premiers médicaments (test)",
    )
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        sys.exit(f"Fichier introuvable : {INPUT_FILE}")

    print(f"Lecture de {INPUT_FILE} ({INPUT_FILE.stat().st_size / 1e9:.2f} Go)...")

    reader_kwargs = dict(
        sep="\t",
        quotechar='"',
        engine="python",
        dtype=str,
        na_filter=False,
        chunksize=CHUNKSIZE,
    )

    records = []
    total_processed = 0

    with tqdm(desc="Médicaments traités", unit=" med") as pbar:
        for chunk in pd.read_csv(INPUT_FILE, **reader_kwargs):
            for _, row in chunk.iterrows():
                records.append(parse_rcp(row["Code_CIS"], row["RCP_html"]))
                total_processed += 1
                pbar.update(1)

                if args.sample and total_processed >= args.sample:
                    break

            if args.sample and total_processed >= args.sample:
                break

    df = pd.DataFrame(records)
    print(f"\n{len(df)} médicaments extraits.")

    # Taux de remplissage par colonne — permet de détecter les ancres manquantes
    fill_rates = (df != "").mean() * 100
    print("\nTaux de remplissage par colonne :")
    for col, rate in fill_rates.items():
        flag = "  ⚠" if rate < 50 else ""
        print(f"  {col:<30} {rate:5.1f}%{flag}")

    if len(df) > 1_000_000:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\nExport CSV (>{1_000_000} lignes) : {OUTPUT_CSV.resolve()}")
    else:
        print(f"\nExport Excel en cours → {OUTPUT_XLSX} ...")
        df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
        print(f"Export terminé : {OUTPUT_XLSX.resolve()}")


if __name__ == "__main__":
    main()
