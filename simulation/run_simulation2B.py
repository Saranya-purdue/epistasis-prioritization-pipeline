from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from simulation2 import (
    simulate_phenotype_with_overlap,
    standardize_genotype_df,
)

GENOTYPE_PATH = "./snps_data.csv"
OUTPUT_DIR = "./simulation2B_outputs"

CAUSAL_SNP_SIZES = [
    15,
    50,
    100,
    250,
]

OVERLAP_PROPORTIONS = [
    0.30,
    0.50,
    0.70,
    0.90,
]

H2_TOTAL = 0.20

# 5% additive / 95% epistatic genetic variance
RHO = 0.05

BASE_SEED = 42


def save_list(
    filepath: Path,
    values: list[str],
) -> None:

    with filepath.open(
        "w",
        encoding="utf-8",
    ) as f:

        for item in values:
            f.write(f"{item}\n")


def main() -> None:

    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load genotype matrix
    # --------------------------------------------------------
    geno_df = pd.read_csv(
        GENOTYPE_PATH,
        index_col=0,
    )

    print(
        f"Loaded genotype matrix: "
        f"{geno_df.shape[0]} samples x "
        f"{geno_df.shape[1]} SNPs"
    )

    # Standardize once for efficiency.
    geno_std = standardize_genotype_df(
        geno_df
    )

    summary_rows = []

    scenario_counter = 0

    # ========================================================
    # Run Simulation 2B
    # ========================================================
    for n_snps in CAUSAL_SNP_SIZES:

        for overlap in OVERLAP_PROPORTIONS:

            scenario_counter += 1

            scenario_seed = (
                BASE_SEED + scenario_counter
            )

            # Simulation 2B:
            # number of true interaction pairs equals
            # the component-specific causal SNP count.
            n_true_pairs = n_snps

            overlap_label = (
                str(overlap)
                .replace(".", "p")
            )

            scenario_name = (
                f"sim2B_n{n_snps}_"
                f"overlap{overlap_label}"
            )

            scenario_dir = (
                output_dir / scenario_name
            )

            scenario_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            print(
                f"\nRunning {scenario_name}"
                f" | seed={scenario_seed}"
                f" | true_pairs={n_true_pairs}"
            )

            # ------------------------------------------------
            # Simulate phenotype
            # ------------------------------------------------
            sim_result = (
                simulate_phenotype_with_overlap(
                    geno_df=geno_std,

                    num_additive=n_snps,
                    num_epistatic=n_snps,

                    overlap_proportion=overlap,

                    h2_total=H2_TOTAL,
                    rho=RHO,

                    seed=scenario_seed,

                    # Key Simulation 2B setting
                    num_epistatic_pairs=n_true_pairs,

                    # Already standardized above
                    standardize_genotypes=False,
                )
            )

            # ------------------------------------------------
            # Save phenotype
            # ------------------------------------------------
            sim_result[
                "phenotype_df"
            ].to_csv(
                scenario_dir / "phenotype.csv"
            )

            # ------------------------------------------------
            # Save causal SNP sets
            # ------------------------------------------------
            save_list(
                scenario_dir /
                "additive_snps.txt",

                sim_result[
                    "additive_snps"
                ],
            )

            save_list(
                scenario_dir /
                "epistatic_snps.txt",

                sim_result[
                    "epistatic_snps"
                ],
            )

            save_list(
                scenario_dir /
                "active_epistatic_snps.txt",

                sim_result[
                    "active_epistatic_snps"
                ],
            )

            save_list(
                scenario_dir /
                "shared_snps.txt",

                sim_result[
                    "shared_snps"
                ],
            )

            # ------------------------------------------------
            # Save additive effect sizes
            # ------------------------------------------------
            beta_add_df = pd.DataFrame(
                {
                    "SNP":
                        sim_result[
                            "additive_snps"
                        ],

                    "beta_add":
                        sim_result[
                            "beta_add"
                        ],
                }
            )

            beta_add_df.to_csv(
                scenario_dir /
                "beta_add.csv",

                index=False,
            )

            # ------------------------------------------------
            # Save true epistatic pairs/effect sizes
            # ------------------------------------------------
            beta_epi_df = pd.DataFrame(
                sim_result[
                    "epistatic_pairs"
                ],
                columns=[
                    "SNP1",
                    "SNP2",
                ],
            )

            beta_epi_df[
                "beta_epi"
            ] = sim_result[
                "beta_epi"
            ]

            beta_epi_df.to_csv(
                scenario_dir /
                "beta_epi.csv",

                index=False,
            )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------
            metadata = {

                "simulation":
                    "Simulation_2B",

                "architecture":
                    "controlled_sparse_epistasis",

                "scenario_name":
                    scenario_name,

                "seed":
                    scenario_seed,

                "num_additive":
                    n_snps,

                "num_epistatic":
                    n_snps,

                "requested_num_true_pairs":
                    n_true_pairs,

                "overlap_proportion":
                    overlap,

                "h2_total":
                    H2_TOTAL,

                "rho":
                    RHO,

                "num_overlap":
                    sim_result[
                        "num_overlap"
                    ],

                "num_unique_add":
                    sim_result[
                        "num_unique_add"
                    ],

                "num_unique_epi":
                    sim_result[
                        "num_unique_epi"
                    ],

                "num_epistatic_pairs":
                    sim_result[
                        "num_epistatic_pairs"
                    ],

                "num_active_epistatic_snps":
                    sim_result[
                        "num_active_epistatic_snps"
                    ],

                "realized_var_add":
                    sim_result[
                        "realized_var_add"
                    ],

                "realized_var_epi":
                    sim_result[
                        "realized_var_epi"
                    ],

                "realized_cov_add_epi":
                    sim_result[
                        "realized_cov_add_epi"
                    ],

                "realized_var_genetic":
                    sim_result[
                        "realized_var_genetic"
                    ],

                "realized_var_noise":
                    sim_result[
                        "realized_var_noise"
                    ],

                "realized_var_phenotype":
                    sim_result[
                        "realized_var_phenotype"
                    ],

                "realized_h2":
                    sim_result[
                        "realized_h2"
                    ],
            }

            with (
                scenario_dir /
                "metadata.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    metadata,
                    f,
                    indent=2,
                )

            summary_rows.append(
                metadata
            )

            print(
                f"Saved {scenario_name}"
                f" | pairs="
                f"{sim_result['num_epistatic_pairs']}"
                f" | active epi SNPs="
                f"{sim_result['num_active_epistatic_snps']}"
                f" | Var(add)="
                f"{sim_result['realized_var_add']:.4f}"
                f" | Var(epi)="
                f"{sim_result['realized_var_epi']:.4f}"
                f" | realized H2="
                f"{sim_result['realized_h2']:.4f}"
            )

    # ========================================================
    # Save overall summary
    # ========================================================
    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        output_dir /
        "simulation2B_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nDone. Summary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()
