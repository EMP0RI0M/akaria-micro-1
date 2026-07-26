import os
import sys
import json
import time
import shutil
import subprocess

STATE_FILE = "experiment_suite_state.json"
REQUIRED_SPACE_GB = 5.0 # Need at least 5 GB free

# Map standard names to what's in train_tinystories.py
# E1 and E2 are added here to satisfy the sequential requirements, 
# even though they might not be implemented in train_tinystories.py yet.
SEQUENCE = ["E0", "Baseline", "E1", "E2", "E3", "E4_0.1", "E5"]

def check_disk_space():
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024 ** 3)
    return free_gb

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_state(state):
    tmp_file = f"{STATE_FILE}.tmp"
    with open(tmp_file, "w") as f:
        json.dump(state, f, indent=4)
    os.replace(tmp_file, STATE_FILE)

def main():
    print("="*60)
    print("INITIALIZING SEQUENTIAL EXPERIMENT SUITE")
    print("="*60)
    
    state = load_state()
    
    # First, run a dry-run or quick parse to see what is implemented
    try:
        # We can extract the contestants from the file directly or we just try to run it.
        # Since we modified train_tinystories.py to take --contestant and exit 1 if not found,
        # we can just rely on the subprocess return code.
        pass
    except Exception as e:
        print(f"Error checking implemented variants: {e}")

    for contestant in SEQUENCE:
        print(f"\n[{contestant}] Checking status...")
        
        c_state = state.get(contestant, {})
        status = c_state.get("status", "pending")
        
        if status == "completed":
            print(f"[{contestant}] Already completed. Skipping.")
            continue
            
        free_gb = check_disk_space()
        if free_gb < REQUIRED_SPACE_GB:
            print(f"CRITICAL: Free disk space ({free_gb:.2f} GB) is below required {REQUIRED_SPACE_GB} GB.")
            print("Stopping suite gracefully to prevent corruption.")
            sys.exit(1)
            
        print("="*60)
        print(f"STARTING {contestant}")
        print("Target: 25,000 steps")
        print("="*60)
        
        c_state["status"] = "running"
        c_state["start_time"] = c_state.get("start_time", time.time())
        state[contestant] = c_state
        save_state(state)
        
        # Run subprocess
        # We do not use DDP/torchrun because it modifies batch semantics and requires 
        # testing that we cannot perform safely in this environment.
        cmd = [sys.executable, "-u", "scripts/train_tinystories.py", "--contestant", contestant]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # We stream output
        process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        process.communicate()
        
        if process.returncode == 0:
            # Assume completed successfully
            print("="*60)
            print(f"{contestant} COMPLETE")
            print("="*60)
            c_state["status"] = "completed"
            c_state["end_time"] = time.time()
            state[contestant] = c_state
            save_state(state)
        elif process.returncode == 1:
            # Usually our custom sys.exit(1) for "Contestant not found"
            print(f"[{contestant}] Not implemented or failed early. Skipping.")
            c_state["status"] = "not_implemented_or_failed"
            state[contestant] = c_state
            save_state(state)
        else:
            print(f"[{contestant}] Crashed with return code {process.returncode}.")
            c_state["status"] = "crashed"
            state[contestant] = c_state
            save_state(state)
            
            print("Stopping suite due to crash.")
            sys.exit(1)

if __name__ == "__main__":
    main()
