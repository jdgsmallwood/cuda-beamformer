import subprocess
import os
import mlflow
import tempfile
import shutil
from dotenv import load_dotenv
import argparse

load_dotenv()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("2025-beamforming-cuda-optimization")


# === Step 1: Ask for user input ===
parser = argparse.ArgumentParser(description="Parse Nsight Compute .ncu-rep file and log to MLflow")

parser.add_argument("run_description", type=str, help="The description of the profiling run")
parser.add_argument("change_description", type=str, help="What did you change?")
parser.add_argument("hypothesis", type=str, help="Hypothesis for the profiling run")
parser.add_argument("nt_session", type=str, help="Which NT node is your session running on? i.e. gina3")

# Parse arguments
args = parser.parse_args()

# === Step 2: Auto-get Git commit ID ===
git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

# === Step 3: Define constants ===
LOCAL_CU_FILE = "beamformer_for_loop.cu"
REMOTE_PATH = "/fred/oz002/jsmallwo/cuda_beamformer/"
REMOTE_CU = f"{REMOTE_PATH}/{LOCAL_CU_FILE}"
REMOTE_EXEC_NAME = "beamformer"
REMOTE_EXEC = f"./{REMOTE_EXEC_NAME}"
REMOTE_HOST = "nt"
PROFILE_OUTPUT = "profile"
LOCAL_OUTPUT_DIR = "./profile_results"

print("push to remote")
# === Step 5: Push .cu to remote ===
subprocess.run([
    "rsync", "-avz", LOCAL_CU_FILE, f"{REMOTE_HOST}:{REMOTE_PATH}"
])

print("compile & run")
# === Step 6: Compile and profile remotely ===
subprocess.run(
#     [
#     "ssh", f"{REMOTE_HOST}", "-t",
#     f"\"ssh {args.nt_session} -t 'cd {REMOTE_PATH} && source setup.sh && nvcc {LOCAL_CU_FILE} -o {REMOTE_EXEC_NAME} && ncu --set full --target-processes all --export {PROFILE_OUTPUT} {REMOTE_EXEC}'\""
# ]
    f"ssh {REMOTE_HOST} -t \"ssh {args.nt_session} -t 'cd {REMOTE_PATH} && source setup.sh && nvcc {LOCAL_CU_FILE} -o {REMOTE_EXEC_NAME} && ncu -f --set full --target-processes all --export {PROFILE_OUTPUT} {REMOTE_EXEC} && ncu --import {PROFILE_OUTPUT}.ncu-rep --csv --page details > {PROFILE_OUTPUT}.csv'\""
    , shell=True)

# === Step 7: Pull back results ===
print("pull back")
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
subprocess.run([
    "rsync", "-avz", f"{REMOTE_HOST}:{REMOTE_PATH}/{PROFILE_OUTPUT}.csv",
    LOCAL_OUTPUT_DIR
])

local_rep_path = os.path.join(LOCAL_OUTPUT_DIR, f"{PROFILE_OUTPUT}.csv")

def extract_metrics_from_csv(csv_text):
    """
    Very basic CSV parser to extract metric name and value pairs.
    This assumes 3-column format: Kernel Name, Metric, Value
    """
    metrics = {}
    lines = csv_text.splitlines()
    for line in lines:
        if line.startswith("#") or "Kernel Name" in line:
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        name = parts[1].strip()
        try:
            value = float(parts[2].strip())
            metrics[name] = value
        except ValueError:
            continue
    return metrics


with mlflow.start_run() as run:
    mlflow.log_param("run_description", args.run_description)
    mlflow.log_param("change_description", args.change_description)
    mlflow.log_param("hypothesis", args.hypothesis)

    
    

    with open(f"{PROFILE_OUTPUT}.csv", "r") as f:
        csv_text = f.read()

    metrics = extract_metrics_from_csv(csv_text)
    for k, v in metrics.items():
        metric_key = f"{k}".replace(" ", "_")
        mlflow.log_metric(metric_key, v)

    mlflow.log_artifact(f"{PROFILE_OUTPUT}.csv", artifact_path=f"parsed_page")

    print(f"✅ MLflow run completed: {run.info.run_id}")