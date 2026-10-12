# External components

Place third-party or main-pipeline dependencies here locally; they are ignored by Git where appropriate.

Recommended layout:

```text
external/
  epigen/              # clone of https://github.com/daisybio/epigen
  vaee_epi_core/       # exact manuscript versions of the five core VAE-Epi scripts
```

The repository scripts do not contain institution-specific absolute paths and accept these components through command-line arguments.
