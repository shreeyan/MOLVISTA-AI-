@echo off
setlocal
REM Batch inference: process a directory of protein/ligand files and compute accuracy
REM Usage: scripts\run_inference_dir.bat <input_dir> [output_dir]

if "%~1"=="" (
  echo Usage: %~nx0 ^<input_dir^> [output_dir]
  exit /b 1
)

set INPUT_DIR=%~1
set OUTPUT_DIR=%~2
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=%ROOT%\EquiBind\data\results\output

set KMP_DUPLICATE_LIB_OK=TRUE
echo Input : %INPUT_DIR%
echo Output: %OUTPUT_DIR%

conda run -n equibind_cpu python EquiBind/run_integrated_prediction.py --input_dir "%INPUT_DIR%" --output "%OUTPUT_DIR%" --verbose
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