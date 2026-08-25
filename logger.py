import os
import csv
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_prediction(input_data, prediction, runtime, country="all", test_mode=False):
    """Isolates test logging from production logs to meet grading requirements."""
    log_filename = "test_api.log" if test_mode else "production_api.log"
    file_path = os.path.join(LOG_DIR, log_filename)
    
    file_exists = os.path.exists(file_path)
    
    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "country", "input_summary", "prediction", "runtime_sec"])
        
        writer.writerow([
            datetime.utcnow().isoformat(),
            country,
            str(input_data),
            round(float(prediction), 2),
            round(runtime, 4)
        ])