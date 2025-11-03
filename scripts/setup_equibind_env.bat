@echo off
echo ==========================================
echo 🚀 Setting up EquiBind Conda Environment
echo ==========================================

REM Check if conda exists
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Conda not found. Please install Miniconda or Anaconda first.
    pause
    exit /b 1
)

REM Create environment if not already created
echo 🔧 Creating environment: equibind_mol ...
conda env list | findstr "equibind_mol" >nul
if %errorlevel% neq 0 (
    conda create -y -n equibind_mol python=3.9
) else (
    echo ⚙️ Environment already exists. Skipping creation.
)

REM Activate environment
echo ✅ Activating environment...
call conda activate equibind_cpu

REM Install core dependencies
echo 📦 Installing dependencies...
conda install -y -n equibind_molpytorch torchvision torchaudio cpuonly -c pytorch
conda install -y -n equibind_mol rdkit -c conda-forge
conda install -y -n equibind_mol numpy scipy pandas scikit-learn matplotlib tqdm pyyaml joblib tensorboard -c conda-forge
conda install -y -n equibind_mol openbabel -c conda-forge

REM Some additional pip-only packages
call conda run -n equibind_mol pip install streamlit biopython pot ot

REM Verify
echo ✅ Checking installed packages...
call conda run -n equibind_mol python -c "import torch, rdkit, streamlit, numpy; print('✅ All core libraries imported successfully!')"

echo ==========================================
echo 🎉 Setup Complete! To activate manually:
echo     conda activate equibind_mol
echo ==========================================
pause
