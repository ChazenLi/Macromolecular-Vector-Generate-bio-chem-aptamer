import logging
import os
from datetime import datetime
from pathlib import Path

class TrainingLogger:
    def __init__(self):
        self._setup_logger()
    
    def _setup_logger(self):
        log_dir = Path(__file__).parents[2] / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            filename=log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M')}.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger()
        
    def log_metrics(self, epoch: int, loss: float):
        self.logger.info(f"Epoch {epoch} | Loss: {loss:.4f}")
        
    def log_error(self, error: Exception):
        self.logger.error(f"Error: {str(error)}", exc_info=True)