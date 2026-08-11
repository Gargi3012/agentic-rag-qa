import json
import logging
import sys
from typing import Dict, Any

class StructuredJsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    If the log record has a `metrics` dictionary, it embeds it as a top-level key.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Extract custom metrics dictionary if attached
        if hasattr(record, "metrics"):
            log_data["metrics"] = record.metrics
            
        return json.dumps(log_data)

def setup_logging():
    """
    Configures the root logger to output in JSON format to stdout.
    """
    logger = logging.getLogger("agentic_rag")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicating logs if handlers already exist
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(console_handler)
        logger.propagate = False
        
    logging.info("Structured JSON logging initialized successfully.")
