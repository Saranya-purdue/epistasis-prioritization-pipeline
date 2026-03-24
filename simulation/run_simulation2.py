from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from simulate2 import simulate_phenotype_sim2


# ----------------------------
# User settings
# ----------------------------
GENOTYPE_PATH = "./geno.csv"
OUTPUT_DIR = "./simulation2_outputs"

CAUSAL_SNP_SIZES = [15, 50, 100, 250]

TOT_H2 = 0.20
RHO = 0.05
BASE_SEED = 42


def save_list(filepath: Path, values: list[str]) -> None:
    with filepath.open("w", encoding="utf-8") as f:
        for item in values:
            f.write(f"{item}\n")


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    geno_df = pd.read_csv(GENOTYPE_PATH, index_col=0)
    print(f"Loaded genotype matrix: {geno_df.shape[0]} samples x {geno_df.shape[1]} SNPs")

    summary_rows = []

    for idx, num_causal_snp in enumerate(CAUSAL_SNP_SIZES, start=1):
        scenario_seed = BASE_SEED + idx
        scenario_name = f"sim2_n{num_causal_snp}"
        scenario_dir = output_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRunning {scenario_name} | seed={scenario_seed}")

        sim_result = simulate_phenotype_sim2(
            geno_df=geno_df,
            num_causal_snp=num_causal_snp,
            tot_h2=TOT_H2,
            rho=RHO,
            seed=scenario_seed,
        )

        # Save phenotype
        sim_result["phenotype_df"].to_csv(scenario_dir / "simulated_phenotype.csv", index=False)

        # Save SNP lists
        save_list(scenario_dir / "causal_snps.txt", sim_result["causal_snps"])
        save_list(scenario_dir / "s1_snps.txt", sim_result["s1_snps"])
        save_list(scenario_dir / "s2_snps.txt", sim_result["s2_snps"])
        save_list(scenario_dir / "s3_snps.txt", sim_result["s3_snps"])

        # Save beta tables
        sim_result["beta_epistatic_df"].to_csv(scenario_dir / "beta_epistatic.csv", index=False)
        sim_result["beta_marginal_df"].to_csv(scenario_dir / "beta_marginal.csv", index=False)

        # Save metadata
        metadata = {
            "scenario_name": scenario_name,
            "seed": scenario_seed,
            "num_causal_snp": num_causal_snp,
            "tot_h2": TOT_H2,
            "rho": RHO,
            "max_pairs_target": sim_result["max_pairs_target"],
            "num_epistatic_pairs": sim_result["num_epistatic_pairs"],
            "s1_len": sim_result["s1_len"],
            "s2_len": sim_result["s2_len"],
            "s3_len": sim_result["s3_len"],
            "realized_var_marginal": sim_result["realized_var_marginal"],
            "realized_var_epistatic": sim_result["realized_var_epistatic"],
            "realized_var_genetic": sim_result["realized_var_genetic"],
            "realized_var_noise": sim_result["realized_var_noise"],
        }

        with (scenario_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        summary_rows.append(metadata)

        print(
            f"Saved {scenario_name} | "
            f"pairs={sim_result['num_epistatic_pairs']} | "
            f"s1={sim_result['s1_len']} | "
            f"s2={sim_result['s2_len']} | "
            f"s3={sim_result['s3_len']}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "simulation2_summary.csv", index=False)

    print(f"\nDone. Summary saved to: {output_dir / 'simulation2_summary.csv'}")


if __name__ == "__main__":
    main()