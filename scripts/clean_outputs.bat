@echo off
setlocal
REM Cleanup script: removes generated outputs and temporary artifacts

echo Cleaning generated outputs...

REM Remove demo plots and results (optional)
if exist d:\equibind\EquiBind\plots_demo (
  rmdir /s /q d:\equibind\EquiBind\plots_demo
  echo Removed plots_demo
)
if exist d:\equibind\EquiBind\results_demo (
  rmdir /s /q d:\equibind\EquiBind\results_demo
  echo Removed results_demo
)

REM Remove current run outputs
if exist d:\equibind\EquiBind\data\results\output (
  rmdir /s /q d:\equibind\EquiBind\data\results\output
  echo Removed data\results\output
)

REM Remove saved prediction dumps (keep checkpoints)
if exist d:\equibind\EquiBind\runs\flexible_self_docking\predictions_RDKitFalse.pt (
  del /q d:\equibind\EquiBind\runs\flexible_self_docking\predictions_RDKitFalse.pt
  echo Removed runs\flexible_self_docking\predictions_RDKitFalse.pt
)

echo Cleanup complete.
endlocal