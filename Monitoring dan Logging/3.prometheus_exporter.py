from __future__ import annotations

import os
import time
from pathlib import Path

import psutil
import requests
from prometheus_client import Gauge, start_http_server


EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "8001"))
API_HEALTH_URL = os.getenv("API_HEALTH_URL", "http://localhost:8000/health")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

CPU_USAGE = Gauge("student_classifier_cpu_usage_percent", "CPU usage percent")
MEMORY_USAGE = Gauge("student_classifier_memory_usage_percent", "Memory usage percent")
DISK_USAGE = Gauge("student_classifier_disk_usage_percent", "Disk usage percent")
API_UP = Gauge("student_classifier_api_up", "Whether inference API health endpoint is reachable")
MODEL_ARTIFACT_READY = Gauge("student_classifier_model_artifact_ready", "Whether model artifacts are present")
SYSTEM_UPTIME_SECONDS = Gauge("student_classifier_system_uptime_seconds", "System uptime in seconds")
SYSTEM_BOOT_TIME_SECONDS = Gauge("student_classifier_system_boot_time_seconds", "System boot time as Unix timestamp")


def model_artifacts_ready() -> bool:
    required = ["model.keras", "scaler.joblib", "feature_columns.json", "label_mapping.json"]
    return all((ARTIFACT_DIR / name).exists() for name in required)


def collect() -> None:
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage("/").percent)
    MODEL_ARTIFACT_READY.set(1 if model_artifacts_ready() else 0)
    boot_time = psutil.boot_time()
    SYSTEM_BOOT_TIME_SECONDS.set(boot_time)
    SYSTEM_UPTIME_SECONDS.set(time.time() - boot_time)
    try:
        response = requests.get(API_HEALTH_URL, timeout=3)
        API_UP.set(1 if response.ok else 0)
    except requests.RequestException:
        API_UP.set(0)


def main() -> None:
    start_http_server(EXPORTER_PORT)
    print(f"Prometheus exporter running on port {EXPORTER_PORT}")
    while True:
        collect()
        time.sleep(5)


if __name__ == "__main__":
    main()
