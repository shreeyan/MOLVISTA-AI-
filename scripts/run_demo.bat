@echo off
setlocal
REM Demo runner: runs integrated prediction and computes accuracy
REM Requires conda and the equibind_cpu env created from environment_cpuonly.yml

set KMP_DUPLICATE_LIB_OK=TRUE

echo === Running EquiBind integrated prediction (CPU) ===
echo Protein: d:\equibind\EquiBind\simple_protein.pdb
echo Ligand : d:\equibind\EquiBind\benzene.sdf
echo Output : d:\equibind\EquiBind\data\results\output

conda run -n equibind_cpu python EquiBind/run_integrated_prediction.py --protein d:\equibind\EquiBind\simple_protein.pdb --ligand d:\equibind\EquiBind\benzene.sdf --output d:\equibind\EquiBind\data\results\output --verbose
if errorlevel 1 (
  echo [ERROR] Integrated prediction failed.
  exit /b 1
)

echo === Computing accuracy summary ===
conda run -n equibind_cpu python EquiBind/analysis/compute_accuracy.py d:\equibind\EquiBind\data\results\output\integrated_predictions.json --out_json d:\equibind\EquiBind\data\results\output\accuracy_summary.json --out_txt d:\equibind\EquiBind\data\results\output\accuracy_summary.txt
if errorlevel 1 (
  echo [ERROR] Accuracy computation failed.
  exit /b 1
)

echo.
echo === Outputs ===
echo Docked SDF: d:\equibind\EquiBind\data\results\output\simple_protein_benzene\docked_ligand.sdf
echo Integrated JSON: d:\equibind\EquiBind\data\results\output\integrated_predictions.json
echo Summary JSON: d:\equibind\EquiBind\data\results\output\accuracy_summary.json
echo Summary TXT : d:\equibind\EquiBind\data\results\output\accuracy_summary.txt
echo.
echo Done.
endlocal