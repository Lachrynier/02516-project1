from loguru import logger
import os
import sys

def setup_logger(model_name: str):
    """Configure loguru logger for console + model-specific file."""
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/{model_name}.log"

    # remove any existing handlers to avoid duplicate logs
    logger.remove()

    # Console output with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>",
        level="INFO",
    )

    # File output (append mode, keep accumulating logs for this model)
    logger.add(
        log_filename,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        enqueue=True,   # safe for multiprocessing
        backtrace=False,
        diagnose=False,
    )

    return logger
