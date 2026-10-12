# Reproducibility notes

## Benchmark A

The analysis notebook evaluates 10, 50 and 100 true pairs under four native EpiGEN architectures. The interaction parameter is calibrated to an epistatic variance fraction of approximately 0.10 with noise SD = 1.0. The common detector benchmark uses the same 2,000-SNP search space for all methods.

## Benchmark B

The end-to-end benchmark starts from a 10,000-SNP input and evaluates phenotype-weighted VAE selection to 2,000 SNPs before RF-SHAP / RF-SHAP-Taylor. The full 10k branch is retained as the comparison. Upstream causal-pair loss is counted in end-to-end recovery; conditional recovery is reported separately to distinguish ranking performance from selection/screening loss.

## Search-space sizes

- 10,000 SNPs: 49,995,000 unordered pairs.
- 2,000 SNPs: 1,999,000 unordered pairs.
- Nominal pair-space reduction: approximately 96.0%.

## Compute

Full 10k exhaustive scans and SHAP interaction analysis can require substantial memory and runtime. The original notebook used a batch scheduler, but scheduler-specific code is deliberately excluded from the public repository. The command-line scripts can be wrapped in any scheduler used by the reviewer or computing environment.
