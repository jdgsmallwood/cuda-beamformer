import subprocess
import os
import mlflow
import tempfile
import shutil
import argparse
import time
import pandas as pd
from loguru import logger


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("2025-beamforming-cuda-optimization")


# === Step 1: Ask for user input ===
parser = argparse.ArgumentParser(
    description="Parse Nsight Compute .ncu-rep file and log to MLflow"
)

parser.add_argument(
    "run_description", type=str, help="The description of the profiling run"
)
parser.add_argument("change_description", type=str, help="What did you change?")
parser.add_argument("hypothesis", type=str, help="Hypothesis for the profiling run")

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
SLURM_FILE_NAME = "submit_job.sh"

logger.info("Initiating push to remote for CUDA file.")
# === Step 4: Push .cu to remote ===
subprocess.run(["rsync", "-avz", LOCAL_CU_FILE, f"{REMOTE_HOST}:{REMOTE_PATH}"])


# === Step 5: Push SLURM job to remote ===
params = {
    "REMOTE_PATH": REMOTE_PATH,
    "LOCAL_CU_FILE": LOCAL_CU_FILE,
    "REMOTE_EXEC_NAME": REMOTE_EXEC_NAME,
    "PROFILE_OUTPUT": PROFILE_OUTPUT,
    "REMOTE_EXEC": REMOTE_EXEC,
}


# Template string for the SLURM shell script
# This will spin up a session with a GPU, compile and profile the code, then create a CSV file for output.
slurm_script = f"""#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=test_%j.txt
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem=16g
#SBATCH --gres=gpu:1

srun cd {params['REMOTE_PATH']} && \\
    source setup.sh && \\
    nvcc {params['LOCAL_CU_FILE']} -o {params['REMOTE_EXEC_NAME']} && \\
    ncu -f --set full --target-processes all --export {params['PROFILE_OUTPUT']} {params['REMOTE_EXEC']} && \\
    ncu --import {params['PROFILE_OUTPUT']}.ncu-rep --csv --page details > {params['PROFILE_OUTPUT']}.csv
"""

# Write to file
with open("submit_job.sh", "w") as f:
    f.write(slurm_script)

logger.info("Slurm script written to 'submit_job.sh'")


# === Step 6: Sync slurm script to server ===
logger.info("Syncing slurm script to remote...")
subprocess.run(["rsync", "-avz", SLURM_FILE_NAME, f"{REMOTE_HOST}:{REMOTE_PATH}"])


# === Step 7: Compile and profile remotely ===
logger.info("Submitting slurm job...")
result = subprocess.run(
    f'ssh {REMOTE_HOST} -t "sbatch {REMOTE_PATH}/submit_job.sh"',
    shell=True,
    capture_output=True,
    text=True,
)

output = result.stdout.strip()

job_id = None
if "Submitted batch job" in output:
    job_id = output.split()[-1]
else:
    raise RuntimeError("Failed to get SLURM job ID.")

# Poll until job finishes
while True:
    cum_time = 0
    check_cmd = f"ssh {REMOTE_HOST} -t 'squeue -j {job_id}'"
    check = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
    if job_id not in check.stdout:
        logger.info(f"Job {job_id} is done!")
        break
    else:
        logger.info(f"Waiting for job {job_id}...")
        time.sleep(10)
        cum_time += 10
        if cum_time >= 10_000:
            raise Exception("Timeout!")


# === Step 8: Pull back results ===
logger.info("Pulling back results...")
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
subprocess.run(
    [
        "rsync",
        "-avz",
        f"{REMOTE_HOST}:{REMOTE_PATH}/{PROFILE_OUTPUT}.csv",
        LOCAL_OUTPUT_DIR,
    ]
)

local_rep_path = os.path.join(LOCAL_OUTPUT_DIR, f"{PROFILE_OUTPUT}.csv")




# === Step 9: Log results to MLFlow ===
def extract_metric(df, metric: str):
    return float(df[df["Metric Name"] == metric]["Metric Value"].iloc[0])


def extract_metrics_from_csv(df):
    """ """
    metrics_config = {
        "theoretical_occupancy": "Theoretical Occupancy",
        "achieved_occupancy": "Achieved Occupancy",
        "l2_hit_rate": "L2 Hit Rate",
        "memory_throughput": "Memory Throughput",
        "compute_throughput": "Compute (SM) Throughput",
        "registers_per_thread": "Registers Per Thread",
        "l1_cache_throughput": "L1/TEX Cache Throughput",
        "l2_cache_throughput": "L2 Cache Throughput",
        "duration": "Duration",
        "theoretical_active_warps_per_sm": "Theoretical Active Warps per SM",
        "achieved_active_warps_per_sm": "Achieved Active Warps Per SM",
    }

    metrics = {k: extract_metric(df, v) for k, v in metrics_config.items()}

    return metrics


def extract_parameters_from_csv(df):

    params = {
        "process_name": df["Process Name"].unique()[0],
        "block_size": df["Block Size"].unique()[0],
        "grid_size": df["Grid Size"].unique()[0],
    }

    return params

logger.info("Starting MLFlow run...")
with mlflow.start_run() as run:
    mlflow.log_param("run_description", args.run_description)
    mlflow.log_param("change_description", args.change_description)
    mlflow.log_param("hypothesis", args.hypothesis)

    profile_path = f"{os.path.join(LOCAL_OUTPUT_DIR, PROFILE_OUTPUT)}.csv"
    data = pd.read_csv(profile_path)
    params = extract_parameters_from_csv(data)
    logger.info("Logging parameters...")
    mlflow.log_params(params)

    logger.info("Logging metrics...")
    metrics = extract_metrics_from_csv(csv_text)
    mlflow.log_metrics(metrics)

    mlflow.log_artifact(profile_path, artifact_path=f"parsed_page")

    logger.info(f"✅ MLflow run completed: {run.info.run_id}")
