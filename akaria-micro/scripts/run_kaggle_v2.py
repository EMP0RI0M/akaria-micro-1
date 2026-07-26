import os
import sys
import subprocess

def main():
    base_dir = "/kaggle/working/akaria-micro-1"
    micro_dir = base_dir + "/akaria-micro"
    moe_dir = "/kaggle/working/looped-moe"

    os.environ["PYTHONPATH"] = micro_dir + ":" + moe_dir
    sys.path.append(micro_dir)
    sys.path.append(moe_dir)

    def run_cmd(cmd):
        print(f"\n>>> Running: {cmd}")
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True, cwd=micro_dir)
        print(result.stdout)
        if result.stderr:
            print(f"Errors:\n{result.stderr}")

    run_cmd("python3 -m pytest tests/test_fluxvm_v2.py -v")
    run_cmd("python3 scripts/sweep_fluxvm_v2.py")

if __name__ == "__main__":
    main()
