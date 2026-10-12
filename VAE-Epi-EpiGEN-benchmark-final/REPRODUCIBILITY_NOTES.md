# Reproducibility notes

## Benchmark A

Benchmark A evaluates 10, 50, and 100 true pairs under four EpiGEN architectures: joint dominant, joint recessive, multiplicative, and exponential. Interaction strength is calibrated to a target epistatic variance fraction of approximately 0.10 with noise SD = 1.0. Detector comparisons are performed within a common 2,000-SNP candidate space.

## Benchmark B

Benchmark B starts from a 10,000-SNP genotype input and evaluates phenotype-weighted VAE prioritization to 2,000 SNPs before downstream RF-SHAP or RF-SHAP-Taylor interaction scoring. Full-10k branches are retained for comparison.

End-to-end recall uses the complete causal-pair denominator. Therefore, causal pairs lost during VAE prioritization remain false negatives. Conditional recall is also reported to separate downstream ranking performance from upstream selection/screening losses.

## Search-space sizes

- 10,000 SNPs: 49,995,000 unordered pairs.
- 2,000 SNPs: 1,999,000 unordered pairs.
- Nominal pair-space reduction: approximately 96.0%.

## Seeds and splits

The public scripts expose random seeds as command-line arguments. RF-SHAP and RF-SHAP-Taylor use a fixed 80/20 train/test split within each prepared benchmark condition. The VAE script stores its fixed train/validation/test indices in the output directory so reruns use the same split.

## External software

EpiGEN, PLINK, MatrixEpistasis, and R are third-party dependencies and are not vendored into this repository. The included wrappers accept user-supplied paths or ordinary executables on `PATH`.

## Compute

Full 10k exhaustive pair scans and SHAP interaction analyses can be computationally intensive. No scheduler-specific submission scripts are required by the repository; each analysis component is an ordinary command-line program.
