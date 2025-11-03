@echo off
setlocal
REM Single inference: process one protein/ligand pair and compute accuracy
REM Usage: scripts\run_inference_single.bat <protein.pdb> <ligand.sdf|mol2> [output_dir]

if "%~2"=="" (
  echo Usage: %~nx0 ^<protein.pdb^> ^<ligand.sdf^|mol2^> [output_dir]
  exit /b 1
)

set PROTEIN=%~1
set LIGAND=%~2
set OUTPUT_DIR=%~3
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=d:\equibind\EquiBind\data\results\output

set KMP_DUPLICATE_LIB_OK=TRUE
echo Protein: %PROTEIN%
echo Ligand : %LIGAND%
echo Output : %OUTPUT_DIR%

conda run -n equibind_cpu python EquiBind/run_integrated_prediction.py --protein "%PROTEIN%" --ligand "%LIGAND%" --output "%OUTPUT_DIR%" --verbose
if errorlevel 1 (
  echo [ERROR] Integrated prediction failed.
  exit /b 1
)

echo === Computing accuracy summary ===
conda run -n equibind_cpu python EquiBind/analysis/compute_accuracy.py "%OUTPUT_DIR%\integrated_predictions.json" --out_json "%OUTPUT_DIR%\accuracy_summary.json" --out_txt "%OUTPUT_DIR%\accuracy_summary.txt"
if errorlevel 1 (
  echo [ERROR] Accuracy computation failed.
  exit /b 1
)

echo Done. Outputs in %OUTPUT_DIR%
endlocal