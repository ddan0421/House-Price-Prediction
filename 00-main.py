import subprocess
import logging
from datetime import datetime
from pytz import timezone

# Set up timezone for logging
def timetz(*args):
    return datetime.now(tz).timetuple()

tz = timezone("America/Los_Angeles")
logging.Formatter.converter = timetz

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("execution.log"),
        logging.StreamHandler()
    ]
)

# List of files to run in sequence
SCRIPTS = [
    "01-data-preparation-contextual-imputation.py",
    "02-data-preparation.py",
    "03-data-preparation-interactions.py",
    "04-model-regressions-svr.py",
    "05-model-ML-trees.py",
    "06-model-ML-stacking.py",
    "19-prediction.py"
]

# Function to execute a Python script
def run_script(script_path):
    logging.info(f"Starting execution of {script_path}")
    try:
        result = subprocess.run(["python", script_path], check=True, capture_output=True, text=True)
        logging.info(f"Output from {script_path}:\n{result.stdout}")
        logging.info(f"Successfully executed {script_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing {script_path}: {e.stderr.strip()}")
        raise  # Rethrow to stop execution if needed

# Main execution
def main():
    logging.info("Starting process...")

    for script in SCRIPTS:
        try:
            run_script(script)
        except Exception as e:
            logging.error(f"Stopping execution due to failure in {script}: {e}")
            return

    logging.info("Process completed.")

if __name__ == "__main__":
    main()
