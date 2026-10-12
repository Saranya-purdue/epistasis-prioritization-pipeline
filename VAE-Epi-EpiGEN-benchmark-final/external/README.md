# External software

## EpiGEN

Clone EpiGEN into this directory (or elsewhere) and pass its location through `--epigen-root`:

```bash
git clone https://github.com/daisybio/epigen.git external/epigen
```

The simulation scripts import `utils.parametrized_model.ParametrizedModel` from that checkout.

## Optional detector software

The complete detector comparison also uses PLINK 1.9, R, MatrixEpistasis, and data.table. The wrappers in this repository accept ordinary command-line paths and do not depend on a particular computing environment.
