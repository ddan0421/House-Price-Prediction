import subprocess
import sys

SCRIPTS = [
    "a0_setup_directories",
    "a1_load_raw_data",
    "a2_load_macroecon",
    "a3_contextual_imputation",
    "a4_advanced_imputation",
    "a5_regression_data_prep",
    "a6_svr_data_prep",
    "a7_knn_data_prep",
    "a8_general_ml_data_prep",
    "a9_catboost_data_prep",
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
