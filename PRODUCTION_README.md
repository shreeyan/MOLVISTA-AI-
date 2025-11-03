# EquiBind Production Usage

This guide covers environment setup, running inference on a dataset, and where outputs are saved.

## 1) Environment Setup (Windows)
- Install Conda (Miniconda/Anaconda).
- In a terminal at `d:\equibind`, run:
  - `scripts\setup_env.bat`

## 2) Run Inference
### Single Pair
- `scripts\run_inference_single.bat d:\equibind\EquiBind\simple_protein.pdb d:\equibind\EquiBind\benzene.sdf`

### Directory of Pairs
- `scripts\run_inference_dir.bat <input_dir> [output_dir]`
- `<input_dir>` contains protein (`.pdb`) and ligand files (`.sdf`/`.mol2`).
- Examples:
  - `scripts\run_inference_dir.bat d:\equibind\EquiBind\data\to_predict\test_complex`
  - `scripts\run_inference_dir.bat d:\equibind\your_dataset d:\equibind\EquiBind\data\results\output`

## 3) Outputs
- `integrated_predictions.json`: per-complex metrics (RMSD, centroid distance, affinity scores)
- `prediction_summary.json`: aggregate info for the run
- `accuracy_summary.json` / `accuracy_summary.txt`: Success@2Å, Success@5Å, mean/median metrics
- Per-complex docked ligand files under `<output_dir>/<complex_name>/docked_ligand.sdf`

## 4) Cleanup
- `scripts\clean_outputs.bat` removes generated outputs and demo artifacts, keeping source and checkpoints.

## Notes
- CPU-only operation is supported via the `equibind_cpu` environment.
- If you see OpenMP runtime warnings, the scripts set `KMP_DUPLICATE_LIB_OK=TRUE` to proceed.