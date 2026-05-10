import subprocess
import sys

SCRIPTS = [
    "a1_regression",
    "a2_svr",
    "a3_knn",
    "a4_trees",
    "a5_xgb",
    "a6_lgbm",
    "a7_catboost",
    "a8_stacking",
]

package = __package__

for script in SCRIPTS:
    module = f"{package}.{script}"
    # flush=True so banners appear before subprocess output when stdout is redirected (non-TTY).
    print(f"\n{'='*60}", flush=True)
    print(f"Running {module}", flush=True)
    print(f"{'='*60}\n", flush=True)
    result = subprocess.run([sys.executable, "-u", "-m", module])
    if result.returncode != 0:
        print(f"\nFAILED: {module} exited with code {result.returncode}", flush=True)
        sys.exit(result.returncode)

print(f"\n{'='*60}", flush=True)
print(f"All {package} scripts completed successfully.", flush=True)
print(f"{'='*60}", flush=True)
