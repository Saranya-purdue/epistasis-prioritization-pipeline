from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from simulate import simulate_phenotype_with_overlap, standardize_genotype_df


# ----------------------------
# User settings
# ----------------------------
GENOTYPE_PATH = "./snps_data.csv"
OUTPUT_DIR = "./simulation1_outputs"

CAUSAL_SNP_SIZES = [15, 50, 100, 250]
OVERLAP_PROPORTIONS = [0.3, 0.5, 0.7, 0.9]

H2_TOTAL = 0.20
RHO = 0.05
BASE_SEED = 42


def save_list(filepath: Path, values: list[str]) -> None:
    with filepath.open("w", encoding="utf-8") as f:
        for item in values:
            f.write(f"{item}\n")


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load genotype matrix
    geno_df = pd.read_csv(GENOTYPE_PATH, index_col=0)
    print(f"Loaded genotype matrix: {geno_df.shape[0]} samples x {geno_df.shape[1]} SNPs")

    # Standardize once here for efficiency
    geno_std = standardize_genotype_df(geno_df)

    summary_rows = []
    scenario_counter = 0

    for n_snps in CAUSAL_SNP_SIZES:
        for overlap in OVERLAP_PROPORTIONS:
            scenario_counter += 1
            scenario_seed = BASE_SEED + scenario_counter

            scenario_name = f"sim1_n{n_snps}_overlap{str(overlap).replace('.', 'p')}"
            scenario_dir = output_dir / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nRunning {scenario_name} | seed={scenario_seed}")

            sim_result = simulate_phenotype_with_overlap(
                geno_df=geno_std,
                num_additive=n_snps,
                num_epistatic=n_snps,
                overlap_proportion=overlap,
                h2_total=H2_TOTAL,
                rho=RHO,
                seed=scenario_seed,
                standardize_genotypes=False,  # already standardized above
            )

            # Save phenotype
            sim_result["phenotype_df"].to_csv(scenario_dir / "phenotype.csv")

            # Save SNP lists
            save_list(scenario_dir / "additive_snps.txt", sim_result["additive_snps"])
            save_list(scenario_dir / "epistatic_snps.txt", sim_result["epistatic_snps"])
            save_list(scenario_dir / "shared_snps.txt", sim_result["shared_snps"])

            # Save additive effects
            beta_add_df = pd.DataFrame({
                "SNP": sim_result["additive_snps"],
                "beta_add": sim_result["beta_add"],
            })
            beta_add_df.to_csv(scenario_dir / "beta_add.csv", index=False)

            # Save epistatic effects
            beta_epi_df = pd.DataFrame(sim_result["epistatic_pairs"], columns=["SNP1", "SNP2"])
            beta_epi_df["beta_epi"] = sim_result["beta_epi"]
            beta_epi_df.to_csv(scenario_dir / "beta_epi.csv", index=False)

            # Save metadata
            metadata = {
                "scenario_name": scenario_name,
                "seed": scenario_seed,
                "num_additive": n_snps,
                "num_epistatic": n_snps,
                "overlap_proportion": overlap,
                "h2_total": H2_TOTAL,
                "rho": RHO,
                "num_overlap": sim_result["num_overlap"],
                "num_unique_add": sim_result["num_unique_add"],
                "num_unique_epi": sim_result["num_unique_epi"],
                "num_epistatic_pairs": sim_result["num_epistatic_pairs"],
                "realized_var_add": sim_result["realized_var_add"],
                "realized_var_epi": sim_result["realized_var_epi"],
                "realized_var_genetic": sim_result["realized_var_genetic"],
                "realized_var_noise": sim_result["realized_var_noise"],
            }

            with (scenario_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            summary_rows.append(metadata)

            print(
                f"Saved {scenario_name} | "
                f"pairs={sim_result['num_epistatic_pairs']} | "
                f"Var(add)={sim_result['realized_var_add']:.4f} | "
                f"Var(epi)={sim_result['realized_var_epi']:.4f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "simulation1_summary.csv", index=False)
    print(f"\nDone. Summary saved to: {output_dir / 'simulation1_summary.csv'}")


if __name__ == "__main__":
    main()