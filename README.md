# Certified Test Selection

Experiment code for scenario-based test selection with count-adaptive calibration and bounds on missed failures.

The certificate controls `Pr(L > epsilon) <= delta` for a fixed library, where `L` is the fraction of all failing tests that are skipped. Scores are fixed before calibration. Stopping uses the observed pass/fail path. Calibration and every retained execution count toward total cost.

## Setup

Python 3.12 was used for the experiments.

```bash
pip install -r requirements.txt
python prepare_inputs.py --archive /path/to/replication-package.zip
python run_checks.py
python run_experiments.py smoke
```

You can use `--source /path/to/extracted-package` instead of `--archive`. Exact decision replay requires the full paper replication package, containing `adaptive_replication/` and `reproducibility/`. That archive is distributed separately and is not downloaded by this repository. Frozen data, model files and generated results stay outside Git. Public data sources are [DeepScenario](https://github.com/Simula-COMPLEX/DeepScenario) and [SensoDat](https://doi.org/10.5281/zenodo.10307479).

## Experiments

```bash
python run_experiments.py adaptive --workers 4
python run_experiments.py baselines --workers 4
python run_experiments.py road
python run_experiments.py duration
```

The paper settings use 500 paired calibration draws, three risk settings and a 30% calibration cap. `adaptive` runs EV-count, EV-score, Target-K, cap certification, checkpoints and paired binomial counterparts. `baselines` runs fixed-size, inverse-sampling, budget and subgroup comparisons. `road` integrates frozen HGB, ITS4SDC and DETOUR scores/orders. `duration` requires the adaptive outputs.

Adaptive decisions are written to `adaptive_replication/results/`, summaries to its `summary/` subfolder. Baseline outputs are in `replication/results/`, road and duration outputs in `adaptive_replication/extension_results/`. The smoke run uses the separate ignored directory `validation/smoke/`.

Optional model and ordering reproduction:

```bash
pip install -r requirements-road.txt
python run_experiments.py refit
python run_experiments.py references --subject road
```

`refit` verifies the frozen source-only HGB/logistic scores. `references` reproduces native selection and ordering comparisons, including official ITS4SDC inference and DETOUR. Raw public-data extraction is available through `simulator_archive/extract_fresh.py`. The openpilot component setup is in [OPENPILOT.md](OPENPILOT.md).

## Code layout

- `adaptive_replication/analysis/`: count-adaptive policies, rank certificates, validation and paired summaries.
- `replication/analysis/`: fixed-size, Target-K, budget and subgroup baselines.
- `certskip/`: execution contract, threshold rules, feature extraction and ITS4SDC adapter.
- `reproducibility/`: score refitting and native ordering experiments.
- `analysis/`: common-road integration and duration accounting.
- `simulator_archive/`: public-data extraction and released openpilot component execution.

Third-party data, models and software retain their upstream licenses. They are not included in this code-only repository. Upstream sources are [ITS4SDC](https://github.com/christianbirchler-org/sdc-testing-competition/tree/main/tools/selectors/icst2025_selectors/ITS4SDC), [DETOUR](https://github.com/cetinkaya/detour) and [openpilot](https://github.com/commaai/openpilot).
