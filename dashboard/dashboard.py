import streamlit as st
import subprocess
import os
import time
import json
import pandas as pd
import py3Dmol

st.set_page_config(layout="wide")
st.title("🧬 EquiBind Dashboard")

# --- Session state initialization ---
if "protein_file" not in st.session_state:
    st.session_state.protein_file = None
if "ligand_file" not in st.session_state:
    st.session_state.ligand_file = None
if "structure_files" not in st.session_state:
    st.session_state.structure_files = []
if "integrated_data" not in st.session_state:
    st.session_state.integrated_data = None
if "summary_data" not in st.session_state:
    st.session_state.summary_data = None
if "inference_done" not in st.session_state:
    st.session_state.inference_done = False

# --- File uploads ---
protein_file = st.file_uploader("Upload Protein (.pdb)", type=["pdb"])
ligand_file = st.file_uploader("Upload Ligand (.sdf, .mol2, .pdb)", type=["sdf", "mol2", "pdb"])

if protein_file:
    st.session_state.protein_file = protein_file
if ligand_file:
    st.session_state.ligand_file = ligand_file

# --- Directories ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
OUTPUT_DIR = os.path.join(BASE_DIR, "EquiBind", "data", "results", "output")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Run Inference Button ---
if st.button("Run Inference"):
    if st.session_state.protein_file and st.session_state.ligand_file:
        st.info("✅ Files uploaded successfully. Preparing input folder...")

        # Save uploaded files
        complex_dir = INPUTS_DIR
        os.makedirs(complex_dir, exist_ok=True)
        protein_path = os.path.join(complex_dir, st.session_state.protein_file.name)
        ligand_path = os.path.join(complex_dir, st.session_state.ligand_file.name)
        with open(protein_path, "wb") as f:
            f.write(st.session_state.protein_file.read())
        with open(ligand_path, "wb") as f:
            f.write(st.session_state.ligand_file.read())

        # Run inference command
        bat_file = os.path.join(SCRIPTS_DIR, "run_inference_dir.bat")
        cmd = f'"{bat_file}" "{complex_dir}" "{OUTPUT_DIR}"'
        st.write("📂 Command:")
        st.code(cmd)

        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            log_output = st.empty()
            logs = ""
            for line in process.stdout:
                logs += line
                log_output.text(logs)
            process.wait()
            log_output.empty()

            if process.returncode == 0:
                st.success("✅ Inference completed successfully!")
                st.session_state.inference_done = True

                # Collect structure files
                st.session_state.structure_files = []
                for root, _, files in os.walk(OUTPUT_DIR):
                    for file in files:
                        if file.endswith((".sdf", ".pdb")):
                            st.session_state.structure_files.append(os.path.join(root, file))

                # Load JSON results
                integrated_path = os.path.join(OUTPUT_DIR, "integrated_predictions.json")
                summary_path = os.path.join(OUTPUT_DIR, "prediction_summary.json")

                if os.path.exists(integrated_path):
                    with open(integrated_path, "r") as f:
                        st.session_state.integrated_data = pd.DataFrame(json.load(f))
                else:
                    st.warning("⚠️ integrated_predictions.json not found.")
                    st.session_state.integrated_data = None

                if os.path.exists(summary_path):
                    with open(summary_path, "r") as f:
                        st.session_state.summary_data = pd.DataFrame(json.load(f))
                else:
                    st.warning("⚠️ prediction_summary.json not found.")
                    st.session_state.summary_data = None

            else:
                st.error("❌ Inference failed. Check logs above.")

        except Exception as e:
            st.error(f"🚨 Error while running inference: {e}")

    else:
        st.error("Please upload both Protein and Ligand files before running inference.")

# --- 3D Visualization (only if inference is done) ---
if st.session_state.inference_done and st.session_state.structure_files:
    st.subheader("🧫 3D Molecular Visualization")
    structure_names = [os.path.relpath(f, OUTPUT_DIR) for f in st.session_state.structure_files]
    selected_file = st.selectbox("Select structure to visualize:", structure_names)

    # Only update viewer on selection change
    if selected_file:
        selected_path = os.path.join(OUTPUT_DIR, selected_file)
        with open(selected_path, "r") as f:
            structure_data = f.read()

        file_type = "sdf" if selected_path.endswith(".sdf") else "pdb"
        viewer = py3Dmol.view(width=700, height=500)
        viewer.addModel(structure_data, file_type)
        viewer.setStyle({"stick": {"colorscheme": "cyanCarbon"}})
        viewer.zoomTo()
        st.components.v1.html(viewer._make_html(), height=520)

# --- Display Predictions ---
if st.session_state.integrated_data is not None:
    st.subheader("📊 Integrated Predictions")
    st.dataframe(st.session_state.integrated_data)

if st.session_state.summary_data is not None:
    st.subheader("📈 Prediction Summary")
    st.dataframe(st.session_state.summary_data)
