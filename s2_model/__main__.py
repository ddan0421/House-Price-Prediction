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
    print(f"\n{'='*60}")
    print(f"Running {module}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, "-m", module])
    if result.returncode != 0:
        print(f"\nFAILED: {module} exited with code {result.returncode}")
        sys.exit(result.returncode)

print(f"\n{'='*60}")
print(f"All {package} scripts completed successfully.")
print(f"{'='*60}")
