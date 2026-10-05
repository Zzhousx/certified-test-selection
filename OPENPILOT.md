# openpilot component experiment

This experiment executes the released longitudinal planner and native MPC solver with a shared numerical vehicle model and synthetic messages. It covers the planning component rather than the complete perception and driving stack.

Pinned releases: v0.9.8 (`fb7b9c0f9420`), v0.9.9 (`8aadf02b2fd9`), v0.10.0 (`c085b8af1943`). The formal library has 10,000 scenarios, four families, a 0.05 s step and a 30 s horizon. A gap below 0.4 m is a failure.

From the repository root, with Docker's Linux engine running:

```bash
python prepare_inputs.py
python simulator_archive/fetch_assets.py
docker build -t certified-selection-openpilot simulator_archive
docker run --rm -v "$PWD/simulator_archive:/work" -v "$PWD/certskip:/work/certskip:ro" -e LD_LIBRARY_PATH=/work/vendor/openpilot/v0.9.9/third_party/acados/lib certified-selection-openpilot python build_openpilot_solver.py
docker run --rm --network none -v "$PWD/simulator_archive:/work" -v "$PWD/certskip:/work/certskip:ro" -e LD_LIBRARY_PATH=/work/vendor/openpilot/v0.9.9/third_party/acados/lib certified-selection-openpilot python run_openpilot.py v0.9.8 formal --workers 3
```

Repeat the final command for v0.9.9 and v0.10.0. Generated specifications and outcomes are in `simulator_archive/data/` and `simulator_archive/results/openpilot/formal/`. `evaluate_openpilot.py` fits historical-version scorers and evaluates the recorded outcomes. Settings are generated from `experiment_protocols.py` by `prepare_inputs.py`. The downloaded source releases and Acados libraries remain in the ignored `vendor/` directory.

Commands above use a Bash-style mount path. In PowerShell, replace `$PWD/...` with the corresponding absolute local directory. Setup downloads public pinned sources. Scenario execution uses the released native solver.
