import sys
from loguru import logger
from app.core.config import settings
import os

def setup_logging():
    logger.remove()

    logger.add(
        sys.stdout, 
        level="INFO",
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    log_path = os.path.join(settings.BASE_DIR, "logs", "cogninpc_{time:YYYY-MM-DD}.log")
    logger.add(
        log_path,
        level="DEBUG",
        rotation="10 MB",
        retention="14 days",
        format="{level: <8} | {name}:{function}:{line} - {message}"
    )

    logger.trace("Logging system (Loguru) has been initialized.")