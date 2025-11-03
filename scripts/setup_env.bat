@echo off
setlocal
REM Setup script: creates and verifies the equibind_cpu environment

where conda >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Conda not found in PATH.
  echo Install Miniconda/Anaconda and reopen terminal.
  exit /b 1
)

REM Check if env exists
for /f "tokens=*" %%E in ('conda env list ^| findstr /r "^equibind_cpu"') do set ENV_FOUND=1
if not defined ENV_FOUND (
  echo Creating conda environment 'equibind_cpu'...
  conda env create -f EquiBind/environment_cpuonly.yml -n equibind_cpu
  if errorlevel 1 (
    echo [ERROR] Failed to create environment.
    exit /b 1
  )
)

echo Verifying PyTorch CPU import...
conda run -n equibind_cpu python -c "import torch; import dgl; print('torch', torch.__version__); import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'; print('dgl', dgl.__version__)" || (
  echo [ERROR] Failed to import torch/dgl in equibind_cpu.
  exit /b 1
)

echo Environment 'equibind_cpu' is ready.
endlocal