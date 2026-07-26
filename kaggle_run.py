import subprocess
import os
import sys

def run_cmd(cmd, env=None, cwd=None):
    print(f"\n>>> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout)
    if res.stderr and res.returncode != 0:
        print(f"Error Output:\n{res.stderr}")
    return res.returncode

print("=== 1. Setting up Environment ===")
run_cmd(["rm", "-rf", "/kaggle/working/looped-moe"])
run_cmd(["rm", "-rf", "/kaggle/working/akaria-micro-1"])

run_cmd(["git", "clone", "https://github.com/epfml/looped-moe.git", "/kaggle/working/looped-moe"])
run_cmd(["git", "-C", "/kaggle/working/looped-moe", "checkout", "833fa17a9f7ad16f9445f8e722d7e4a25323f885"])

run_cmd(["git", "clone", "https://github.com/EMP0RI0M/akaria-micro-1.git", "/kaggle/working/akaria-micro-1"])
run_cmd([sys.executable, "-m", "pip", "install", "pytest"])

print("\n=== 2. Running FluxVM Tests ===")
env = os.environ.copy()
env["PYTHONPATH"] = "/kaggle/working/akaria-micro-1/akaria-micro:/kaggle/working/looped-moe"
cwd = "/kaggle/working/akaria-micro-1/akaria-micro"

run_cmd([sys.executable, "-m", "pytest", "-v", "tests/test_fluxvm.py"], env=env, cwd=cwd)

print("\n=== 3. Running Parameter Tying Tests ===")
run_cmd([sys.executable, "-m", "pytest", "-v", "tests/test_parameter_tying.py"], env=env, cwd=cwd)

print("\n=== 4. Running A-E Ablation Harness ===")
run_cmd([sys.executable, "scripts/run_ablations.py"], env=env, cwd=cwd)

print("\n=== 5. Sweeping FluxVM Controller ===")
run_cmd([sys.executable, "scripts/sweep_fluxvm.py"], env=env, cwd=cwd)
