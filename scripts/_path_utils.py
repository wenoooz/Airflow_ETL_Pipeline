"""Path helpers for Windows-safe artifact directories (run_id often contains ':' from Airflow)."""


def safe_run_id(run_id: str) -> str:
    """Replace colons so the name is valid as a folder on Windows. Linux allows ':' but Windows does not."""
    return run_id.replace(":", "-")
