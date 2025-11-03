import os
import shutil
import subprocess
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
CONFIG_PATH = "equibind/configs_clean/inference.yml"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/run_inference/")
async def run_inference(
    protein: UploadFile,
    ligand: UploadFile
):
    # Save uploaded files
    user_dir = os.path.join(UPLOAD_DIR, protein.filename.split(".")[0])
    os.makedirs(user_dir, exist_ok=True)

    protein_path = os.path.join(user_dir, protein.filename)
    ligand_path = os.path.join(user_dir, ligand.filename)

    with open(protein_path, "wb") as f:
        shutil.copyfileobj(protein.file, f)
    with open(ligand_path, "wb") as f:
        shutil.copyfileobj(ligand.file, f)

    # Prepare and run inference command
    cmd = [
        "python", "equibind/inference.py",
        f"--config={CONFIG_PATH}"
    ]

    # (Optional) dynamically modify config with inference_path/output_directory
    # You can instead edit the YAML template before running

    env = os.environ.copy()
    env["INFERENCE_PATH"] = user_dir
    env["OUTPUT_DIR"] = RESULTS_DIR

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    # Find result files (.sdf or .pt)
    output_files = [
        os.path.join(RESULTS_DIR, f)
        for f in os.listdir(RESULTS_DIR)
        if f.endswith(".sdf") or f.endswith(".pt")
    ]

    if not output_files:
        return JSONResponse({"error": "No output generated"}, status_code=500)

    return FileResponse(output_files[0], filename=os.path.basename(output_files[0]))
