"""
Shared configuration and helpers for the intervener-complexity analysis
scripts.

All scripts default to reading/writing under a local ./data and ./outputs
directory so they run anywhere -- no Google Drive or Colab required. If a
script IS run inside Google Colab, it will automatically offer to mount
Drive and use /content/drive/MyDrive/Course_Project instead; everywhere
else this is skipped automatically.
"""
import os

DATA_DIR = os.environ.get("INTERVENER_DATA_DIR", "data")
OUTPUT_DIR = os.environ.get("INTERVENER_OUTPUT_DIR", "outputs")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def maybe_mount_drive(drive_path=None):
    """
    Only mounts Google Drive if actually running inside Google Colab.
    Returns the effective data directory to use (local ./data otherwise).
    """
    try:
        from google.colab import drive
    except ImportError:
        return ensure_dir(DATA_DIR)

    if drive_path is None:
        drive_path = "/content/drive/MyDrive/Course_Project"
    drive.mount("/content/drive")
    return ensure_dir(drive_path)
