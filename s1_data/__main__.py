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
