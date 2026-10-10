# LOCKED-CROSS reproducibility materials

Code and selected verification outputs accompanying the manuscript “LOCKED-CROSS: a coverage-aware framework for validating cancer drug-response models across screening platforms.”

Author: Zhenyu Wang, Xinyang Normal University.

## Reproduce

Use Python 3.11 or later. Create an isolated environment and install the pinned environment with `python -m pip install -r requirements-lock.txt`. Install this package with `python -m pip install -e .`.

The original study inputs are third-party resources and are not redistributed here. Obtain the cited CoderData 2.2.1 release and the GDSCv1/NCI60 holdout inputs under their respective source terms, then follow `docs/confirmatory_protocol_v1.md` and `docs/evaluation_framework.md`. The scripts use paths under a local `data/` directory; inspect each script's input contract before running.

Selected aggregate verification outputs are supplied in `results/repair_historical_population/` and `results/holdout_confirmation_seeded/`. Figures and tables include source CSVs; generated artwork is omitted.

Run implementation tests with `python tests/run_tests.py`. Some tests require optional dependencies listed in `pyproject.toml`, including PyTorch and LightGBM.

This repository contains code and reproducibility materials only. It does not provide clinical recommendations or a universally transferable predictor. No website is required. Protocol deviations and reproducibility limitations are documented in `docs/protocol_deviations.md` and `docs/reproducibility_audit.md`.
