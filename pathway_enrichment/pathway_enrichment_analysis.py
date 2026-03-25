from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


DEFAULT_LIBRARIES = [
    "KEGG_2021_Human",
    "GO_Biological_Process_2021",
    "WikiPathway_2021_Human",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def safe_slug(text: str) -> str:
    out = str(text).strip().lower()
    for ch in [" ", "+", "/", "\\", ":", ";", ",", "(", ")", "[", "]"]:
        out = out.replace(ch, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def split_gene_field(value) -> list[str]:
    """
    Handle gene annotations like:
      "GENE1;GENE2"
      nan
      ""
    """
    if pd.isna(value):
        return []
    genes = [g.strip() for g in str(value).split(";") if g.strip()]
    return genes


def load_and_prepare_data_from_df(
    df: pd.DataFrame,
    snp1_gene_col: str = "snp1_gene",
    snp2_gene_col: str = "snp2_gene",
) -> tuple[pd.DataFrame, list[str], list[tuple[str, str]]]:
    """
    Extract unique genes and unique gene pairs from the mapped-gene pair table.
    """
    all_genes: set[str] = set()
    gene_pairs: set[tuple[str, str]] = set()

    for _, row in df.iterrows():
        genes1 = split_gene_field(row.get(snp1_gene_col))
        genes2 = split_gene_field(row.get(snp2_gene_col))

        for g1 in genes1:
            for g2 in genes2:
                if g1 and g2 and g1 != g2:
                    all_genes.add(g1)
                    all_genes.add(g2)
                    gene_pairs.add(tuple(sorted((g1, g2))))

    return df.copy(), sorted(all_genes), sorted(gene_pairs)


def enrichr_analysis(
    gene_list: list[str],
    gene_set_libraries: Iterable[str],
    timeout_sec: int = 60,
    sleep_sec: float = 0.2,
) -> dict[str, list]:
    if not gene_list:
        return {}

    enrichr_add_url = "https://maayanlab.cloud/Enrichr/addList"
    genes_str = "\n".join(gene_list)

    payload = {"list": (None, genes_str)}
    response = requests.post(enrichr_add_url, files=payload, timeout=timeout_sec)
    response.raise_for_status()

    data = response.json()
    user_list_id = data["userListId"]

    results: dict[str, list] = {}
    for library in gene_set_libraries:
        enrich_url = (
            f"https://maayanlab.cloud/Enrichr/enrich?"
            f"userListId={user_list_id}&backgroundType={library}"
        )
        r = requests.get(enrich_url, timeout=timeout_sec)
        if not r.ok:
            print(f"  - Warning: Enrichr request failed for {library}: {r.text}")
            continue
        lib_data = r.json()
        if library in lib_data:
            results[library] = lib_data[library]
        time.sleep(sleep_sec)

    return results


def parse_enrichr_results(enrichr_results: dict[str, list], top_n: int = 20) -> pd.DataFrame:
    rows = []

    for library, terms in enrichr_results.items():
        for term in terms[:top_n]:
            # Expected term structure:
            # [rank, term_name, pvalue, zscore, combined_score, genes, adj_p]
            genes_field = term[5]
            if isinstance(genes_field, list):
                genes_joined = ";".join(genes_field)
            else:
                genes_joined = str(genes_field)

            rows.append(
                {
                    "Library": library,
                    "Term": term[1],
                    "P-value": term[2],
                    "Adjusted P-value": term[6],
                    "Genes": genes_joined,
                    "Combined Score": term[4],
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["Library", "Term", "P-value", "Adjusted P-value", "Genes", "Combined Score"]
        )

    df = pd.DataFrame(rows).sort_values("Adjusted P-value").reset_index(drop=True)
    return df


def plot_enrichment_results(
    enrichment_df: pd.DataFrame,
    output_file: str | Path,
    top_n: int = 10,
) -> None:
    if enrichment_df.empty:
        print("  - No enrichment results to plot.")
        return

    top_terms = enrichment_df.head(top_n).copy()
    x_vals = -np.log10(top_terms["P-value"].replace(0, np.nan))

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(top_terms))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_terms)))

    ax.barh(y_pos, x_vals, color=colors)

    labels = [
        term if len(term) <= 60 else term[:57] + "..."
        for term in top_terms["Term"]
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("-log10(P-value)", fontsize=12)
    ax.set_title("Top Enriched Pathways / GO Terms", fontsize=14, fontweight="bold")

    ax.axvline(-np.log10(0.05), color="red", linestyle="--", label="p = 0.05")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_string_interactions(
    gene_list: list[str],
    species: int = 9606,
    score_threshold: float = 400,
    timeout_sec: int = 60,
) -> pd.DataFrame:
    if not gene_list:
        return pd.DataFrame(columns=["gene1", "gene2", "score"])

    string_api_url = "https://string-db.org/api"
    output_format = "tsv-no-header"
    method = "network"
    request_url = f"{string_api_url}/{output_format}/{method}"

    params = {
        "identifiers": "%0d".join(gene_list),
        "species": species,
        "caller_identity": "pathway_enrichment_module",
    }

    r = requests.post(request_url, data=params, timeout=timeout_sec)
    if not r.ok:
        print("  - STRING request failed:", r.text)
        return pd.DataFrame(columns=["gene1", "gene2", "score"])

    interactions = []
    for line in r.text.strip().split("\n"):
        cols = line.split("\t")
        if len(cols) >= 6:
            gene1 = cols[2]
            gene2 = cols[3]
            score = float(cols[5])
            interactions.append({"gene1": gene1, "gene2": gene2, "score": score})

    df = pd.DataFrame(interactions)
    if df.empty:
        return df

    df = df[df["score"] >= score_threshold].reset_index(drop=True)
    return df


def analyze_gene_pair_interactions(
    gene_pairs: list[tuple[str, str]],
    string_interactions: pd.DataFrame,
) -> pd.DataFrame:
    if string_interactions.empty:
        return pd.DataFrame(
            columns=[
                "gene1",
                "gene2",
                "direct_interaction",
                "common_neighbors",
                "common_neighbor_genes",
            ]
        )

    interaction_dict: dict[str, set[str]] = defaultdict(set)
    for _, row in string_interactions.iterrows():
        interaction_dict[row["gene1"]].add(row["gene2"])
        interaction_dict[row["gene2"]].add(row["gene1"])

    rows = []
    for g1, g2 in gene_pairs:
        neighbors_g1 = interaction_dict.get(g1, set())
        neighbors_g2 = interaction_dict.get(g2, set())
        direct = g2 in neighbors_g1
        common = neighbors_g1.intersection(neighbors_g2)

        rows.append(
            {
                "gene1": g1,
                "gene2": g2,
                "direct_interaction": direct,
                "common_neighbors": len(common),
                "common_neighbor_genes": ";".join(sorted(common)) if common else "",
            }
        )

    return pd.DataFrame(rows)


def load_gwas_catalog_tsv(gwas_tsv_path: str | Path, gene_col: str = "gene") -> pd.DataFrame:
    df = pd.read_csv(gwas_tsv_path)
    if gene_col not in df.columns:
        raise ValueError(
            f"GWAS TSV must contain column '{gene_col}'. "
            f"Columns found: {list(df.columns)}"
        )
    return df


def run_complete_analysis_df(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    snp1_gene_col: str = "snp1_gene",
    snp2_gene_col: str = "snp2_gene",
    gene_set_libraries: Iterable[str] = DEFAULT_LIBRARIES,
    gwas_tsv_path: str | Path | None = None,
    gwas_gene_col: str = "gene",
    string_species: int = 9606,
    string_score_threshold: float = 400,
    top_n_enrichr: int = 20,
    top_n_plot: int = 10,
    request_timeout: int = 60,
) -> dict:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    print("\n[1] Loading and preparing data...")
    _, all_genes, gene_pairs = load_and_prepare_data_from_df(
        df,
        snp1_gene_col=snp1_gene_col,
        snp2_gene_col=snp2_gene_col,
    )
    print(f"   - Unique genes: {len(all_genes)}")
    print(f"   - Unique gene pairs: {len(gene_pairs)}")

    enrichment_df = pd.DataFrame()
    string_interactions = pd.DataFrame()
    pair_interactions = pd.DataFrame()
    gwas_results = pd.DataFrame()
    n_direct = 0

    # 2. Enrichr
    print("\n[2] Running pathway enrichment (Enrichr)...")
    try:
        enrichr_results = enrichr_analysis(
            all_genes,
            gene_set_libraries=gene_set_libraries,
            timeout_sec=request_timeout,
        )
        enrichment_df = parse_enrichr_results(enrichr_results, top_n=top_n_enrichr)
        enrichment_df.to_csv(output_dir / "pathway_enrichment.csv", index=False)
        plot_enrichment_results(
            enrichment_df,
            output_file=output_dir / "enrichment_plot.png",
            top_n=top_n_plot,
        )
        if not enrichment_df.empty:
            print(
                f"   - {len(enrichment_df)} terms returned "
                f"(top adjusted p term: {enrichment_df.iloc[0]['Term']})"
            )
        else:
            print("   - No enrichment terms returned.")
    except Exception as e:
        print(f"   - Error in enrichment analysis: {e}")

    # 3. STRING
    print("\n[3] Analyzing protein-protein interactions (STRING)...")
    try:
        string_interactions = get_string_interactions(
            all_genes,
            species=string_species,
            score_threshold=string_score_threshold,
            timeout_sec=request_timeout,
        )
        string_interactions.to_csv(output_dir / "string_interactions.csv", index=False)

        pair_interactions = analyze_gene_pair_interactions(gene_pairs, string_interactions)
        pair_interactions.to_csv(output_dir / "gene_pair_interactions.csv", index=False)

        if not pair_interactions.empty:
            n_direct = int(pair_interactions["direct_interaction"].sum())

        print(f"   - STRING edges above threshold: {len(string_interactions)}")
        print(f"   - Gene pairs with direct PPI: {n_direct}/{len(gene_pairs)}")
    except Exception as e:
        print(f"   - Error in PPI analysis: {e}")
        n_direct = 0

    # 4. GWAS TSV overlap
    print("\n[4] Integrating with GWAS TSV...")
    if gwas_tsv_path is None:
        print("   - No gwas_tsv_path provided; skipping GWAS overlap.")
    else:
        try:
            gwas_all = load_gwas_catalog_tsv(gwas_tsv_path, gene_col=gwas_gene_col)

            if gwas_all.empty:
                print("   - GWAS TSV is empty.")
            else:
                gwas_results = gwas_all[gwas_all[gwas_gene_col].isin(all_genes)].copy()

                if not gwas_results.empty:
                    out_path = output_dir / "gwas_catalog_genes_from_tsv.csv"
                    gwas_results.to_csv(out_path, index=False)
                    print(
                        f"   - {len(gwas_all)} total GWAS TSV rows"
                        f"\n   - {len(gwas_results)} overlapping rows"
                        f"\n   - {gwas_results[gwas_gene_col].nunique()} overlapping genes"
                    )
                else:
                    print("   - No overlap between TSV genes and this gene set.")
        except Exception as e:
            print(f"   - Error reading GWAS TSV: {e}")

    summary = {
        "Total genes": len(all_genes),
        "Total gene pairs": len(gene_pairs),
        "Enriched pathways (Adjusted P < 0.05)": int(
            (enrichment_df["Adjusted P-value"] < 0.05).sum()
        ) if not enrichment_df.empty else 0,
        "Gene pairs with direct PPI": int(n_direct),
        "Genes with GWAS hits": int(gwas_results[gwas_gene_col].nunique()) if not gwas_results.empty else 0,
    }

    pd.DataFrame([summary]).to_csv(output_dir / "analysis_summary.csv", index=False)

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved under: {output_dir}")

    return {
        "enrichment": enrichment_df,
        "ppi": pair_interactions,
        "gwas": gwas_results,
        "summary": summary,
        "all_genes": all_genes,
        "gene_pairs": gene_pairs,
    }


def run_batch_analysis(
    *,
    intra_pairs_csv: str | Path | None,
    inter_pairs_csv: str | Path | None,
    output_dir: str | Path,
    classes: list[str],
    class_col: str = "class",
    snp1_gene_col: str = "snp1_gene",
    snp2_gene_col: str = "snp2_gene",
    gwas_tsv_path: str | Path | None = None,
    gwas_gene_col: str = "gene",
    gene_set_libraries: Iterable[str] = DEFAULT_LIBRARIES,
    string_species: int = 9606,
    string_score_threshold: float = 400,
    top_n_enrichr: int = 20,
    top_n_plot: int = 10,
    request_timeout: int = 60,
    run_all_rows: bool = False,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    summary_rows = []
    datasets = {}

    if intra_pairs_csv is not None:
        datasets["intra"] = pd.read_csv(intra_pairs_csv)
    if inter_pairs_csv is not None:
        datasets["inter"] = pd.read_csv(inter_pairs_csv)

    if not datasets:
        raise ValueError("Provide at least one of --intra_pairs_csv or --inter_pairs_csv.")

    for pair_type, df in datasets.items():
        if run_all_rows:
            subset = df.copy()
            combo_dir = output_dir / pair_type / "all_rows"
            result = run_complete_analysis_df(
                subset,
                combo_dir,
                snp1_gene_col=snp1_gene_col,
                snp2_gene_col=snp2_gene_col,
                gene_set_libraries=gene_set_libraries,
                gwas_tsv_path=gwas_tsv_path,
                gwas_gene_col=gwas_gene_col,
                string_species=string_species,
                string_score_threshold=string_score_threshold,
                top_n_enrichr=top_n_enrichr,
                top_n_plot=top_n_plot,
                request_timeout=request_timeout,
            )
            summary_rows.append(
                {
                    "pair_type": pair_type,
                    "class": "ALL_ROWS",
                    "n_rows": len(subset),
                    "n_genes": len(result["all_genes"]),
                    "n_gene_pairs": len(result["gene_pairs"]),
                    **result["summary"],
                }
            )

        if class_col not in df.columns:
            print(f"Skipping class split for {pair_type}: column '{class_col}' not found.")
            continue

        for cls in classes:
            subset = df[df[class_col] == cls].copy()
            combo_dir = output_dir / pair_type / safe_slug(cls)
            result = run_complete_analysis_df(
                subset,
                combo_dir,
                snp1_gene_col=snp1_gene_col,
                snp2_gene_col=snp2_gene_col,
                gene_set_libraries=gene_set_libraries,
                gwas_tsv_path=gwas_tsv_path,
                gwas_gene_col=gwas_gene_col,
                string_species=string_species,
                string_score_threshold=string_score_threshold,
                top_n_enrichr=top_n_enrichr,
                top_n_plot=top_n_plot,
                request_timeout=request_timeout,
            )
            summary_rows.append(
                {
                    "pair_type": pair_type,
                    "class": cls,
                    "n_rows": len(subset),
                    "n_genes": len(result["all_genes"]),
                    "n_gene_pairs": len(result["gene_pairs"]),
                    **result["summary"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "master_analysis_summary.csv", index=False)
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pathway enrichment, STRING PPI, and GWAS overlap analysis for intra/inter SNP-pair classes."
    )

    parser.add_argument("--intra_pairs_csv", default=None)
    parser.add_argument("--inter_pairs_csv", default=None)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--class_col", default="class")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["ML + GWAS + INT", "ML + GWAS", "ML + INT"],
        help="Classes to analyze separately.",
    )
    parser.add_argument("--run_all_rows", action="store_true")

    parser.add_argument("--snp1_gene_col", default="snp1_gene")
    parser.add_argument("--snp2_gene_col", default="snp2_gene")

    parser.add_argument("--gwas_tsv_path", default=None)
    parser.add_argument("--gwas_gene_col", default="gene")

    default="gene")

    parser.add_argument("--gene_set_libraries", nargs="+", default=DEFAULT_LIBRARIES)
    parser.add_argument("--string_species", type=int, default=9606)
    parser.add_argument("--string_score_threshold", type=float, default=400)
    parser.add_argument("--top_n_enrichr", type=int, default=20)
    parser.add_argument("--top_n_plot", type=int, default=10)
    parser.add_argument("--request_timeout", type=int, default=60)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_df = run_batch_analysis(
        intra_pairs_csv=args.intra_pairs_csv,
        inter_pairs_csv=args.inter_pairs_csv,
        output_dir=args.output_dir,
        classes=args.classes,
        class_col=args.class_col,
        snp1_gene_col=args.snp1_gene_col,
        snp2_gene_col=args.snp2_gene_col,
        gwas_tsv_path=args.gwas_tsv_path,
        gwas_gene_col=args.gwas_gene_col,
        gene_set_libraries=args.gene_set_libraries,
        string_species=args.string_species,
        string_score_threshold=args.string_score_threshold,
        top_n_enrichr=args.top_n_enrichr,
        top_n_plot=args.top_n_plot,
        request_timeout=args.request_timeout,
        run_all_rows=args.run_all_rows,
    )

    print("\nBatch summary:")
    print(summary_df)


if __name__ == "__main__":
    main()
