# EquiBind Demo Video Guide

This guide outlines what to show in the video: how to run the model, the inputs used, and where outputs are saved.

## Prerequisites
- `conda` installed
- Environment created: `conda env create -f EquiBind/environment_cpuonly.yml -n equibind_cpu`

## Inputs Used
- Protein: `d:\equibind\EquiBind\simple_protein.pdb`
- Ligand: `d:\equibind\EquiBind\benzene.sdf`
- Output directory: `d:\equibind\EquiBind\data\results\output`

## Commands to Run
1) Run integrated prediction (CPU):
```
set KMP_DUPLICATE_LIB_OK=TRUE
conda run -n equibind_cpu python EquiBind/run_integrated_prediction.py --protein d:\equibind\EquiBind\simple_protein.pdb --ligand d:\equibind\EquiBind\benzene.sdf --output d:\equibind\EquiBind\data\results\output --verbose
```
2) Compute accuracy from the generated JSON:
```
conda run -n equibind_cpu python EquiBind/analysis/compute_accuracy.py d:\equibind\EquiBind\data\results\output\integrated_predictions.json --out_json d:\equibind\EquiBind\data\results\output\accuracy_summary.json --out_txt d:\equibind\EquiBind\data\results\output\accuracy_summary.txt
```

Alternatively, run everything with:
```
scripts\run_demo.bat
```

## What to Show On Screen
- Show the input files in Explorer:
  - `simple_protein.pdb`
  - `benzene.sdf`
- Show the exact command(s) executed.
- Show the console output (device=cpu, processing, RMSD and centroid distance).
- Show the generated outputs in Explorer:
  - `d:\equibind\EquiBind\data\results\output\simple_protein_benzene\docked_ligand.sdf`
  - `d:\equibind\EquiBind\data\results\output\integrated_predictions.json`
  - `d:\equibind\EquiBind\data\results\output\prediction_summary.json`
  - `d:\equibind\EquiBind\data\results\output\accuracy_summary.json`
  - `d:\equibind\EquiBind\data\results\output\accuracy_summary.txt`

## Expected Results (from demo)
- Success@2 Å: `0.0%` (0/1)
- Success@5 Å: `100.0%` (1/1)
- RMSD: `4.698 Å`
- Centroid distance: `4.543 Å`

## Recording Tips
- Use Windows Xbox Game Bar (`Win+G`) or any screen recorder.
- Record at 1080p, 60fps if possible.
- Keep terminal font large for readability.
- Narrate: inputs, command, outputs location, and summary.

## Cleanup
- To clean generated artifacts between takes:
```
scripts\clean_outputs.bat
```