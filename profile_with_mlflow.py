import subprocess
import os
import mlflow
import tempfile
import shutil
import argparse
import time
import pandas as pd
from loguru import logger
from datetime import datetime
from pytz import timezone


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    except subprocess.CalledProcessError:
        return "unknown"

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
DT_SUFFIX = datetime.now(tz=timezone('Australia/Sydney')).strftime('%Y%m%d_%H%M')
PROFILE_OUTPUT = f"profile_{DT_SUFFIX}"
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
    "JOB_OUTPUT_FILE_NAME": f"profile_log_{DT_SUFFIX}.txt",
}


# Template string for the SLURM shell script
# This will spin up a session with a GPU, compile and profile the code, then create a CSV file for output.
slurm_script = f"""#!/bin/bash
#
#SBATCH --job-name=profile
#SBATCH --output={params['JOB_OUTPUT_FILE_NAME']}
#
#SBATCH --ntasks=1
#SBATCH --time=02:30
#SBATCH --mem=16g
#SBATCH --gres=gpu:1

srun cd {params['REMOTE_PATH']} && \\
    source setup.sh && \\
    nvcc --gpu-architecture=sm_80 -O3 --use_fast_math {params['LOCAL_CU_FILE']} -o {params['REMOTE_EXEC_NAME']}  && \\
    ncu -f --set full --target-processes all --export {params['PROFILE_OUTPUT']} {params['REMOTE_EXEC']} && \\
    ncu --import {params['PROFILE_OUTPUT']}.ncu-rep --csv --page details > {params['PROFILE_OUTPUT']}.csv
"""

slurm_script_debug = f"""#!/bin/bash
#
#SBATCH --job-name=profile
#SBATCH --output={params['JOB_OUTPUT_FILE_NAME']}
#
#SBATCH --ntasks=1
#SBATCH --time=02:30
#SBATCH --mem=16g
#SBATCH --gres=gpu:1

srun cd {params['REMOTE_PATH']} && \\
    source setup.sh && \\
    nvcc -G -g {params['LOCAL_CU_FILE']} -o {params['REMOTE_EXEC_NAME']}  && \\
    ncu -f --source-file=all --source-line=all --set full --target-processes all --export {params['PROFILE_OUTPUT']} {params['REMOTE_EXEC']} && \\
    ncu --import {params['PROFILE_OUTPUT']}.ncu-rep --csv --page details > {params['PROFILE_OUTPUT']}.csv
"""

# Write to file
with open("submit_job.sh", "w") as f:
    f.write(slurm_script_debug)

logger.info("Slurm script written to 'submit_job.sh'")


# === Step 6: Sync slurm script to server ===
logger.info("Syncing slurm script to remote...")
subprocess.run(["rsync", "-avz", SLURM_FILE_NAME, f"{REMOTE_HOST}:{REMOTE_PATH}"])


# === Step 7: Compile and profile remotely ===
logger.info("Submitting slurm job...")
subprocess.run(
    f'ssh {REMOTE_HOST} -t "cd {REMOTE_PATH} && sbatch -W submit_job.sh"',
    shell=True,
)


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

subprocess.run(
    [
        "rsync",
        "-avz",
        f"{REMOTE_HOST}:{REMOTE_PATH}/{PROFILE_OUTPUT}.ncu-rep",
        LOCAL_OUTPUT_DIR,
    ]
)

subprocess.run(
    [
        "rsync",
        "-avz",
        f"{REMOTE_HOST}:{REMOTE_PATH}/{params['JOB_OUTPUT_FILE_NAME']}",
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
description = f"""
**Description**
{args.run_description}

**Changes Made**
{args.change_description}

**Hypothesis**
{args.hypothesis}
"""


with mlflow.start_run(description=description) as run:
    profile_path = f"{os.path.join(LOCAL_OUTPUT_DIR, PROFILE_OUTPUT)}.csv"
    data = pd.read_csv(profile_path)
    model_params = extract_parameters_from_csv(data)
    logger.info("Logging parameters...")
    mlflow.log_params(model_params)
    mlflow.log_param("git_commit_hash", get_git_commit())

    logger.info("Logging metrics...")
    metrics = extract_metrics_from_csv(data)
    mlflow.log_metrics(metrics)

    mlflow.log_artifact(profile_path)
    # log the original ncu-rep file as well.
    mlflow.log_artifact(profile_path.replace('.csv', '.ncu-rep'))
    mlflow.log_artifact(LOCAL_CU_FILE)
    mlflow.log_artifact(os.path.join(LOCAL_OUTPUT_DIR, params['JOB_OUTPUT_FILE_NAME']))

    logger.info(f"✅ MLflow run completed: {run.info.run_id}")
