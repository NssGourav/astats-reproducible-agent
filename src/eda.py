import pandas as pd
from src.logger import logger

class EDA:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Loads CSV and logs basic diagnostics."""
        logger.log_step("LOAD", f"Loading dataset from {self.file_path}")
        try:
            self.df = pd.read_csv(self.file_path)
            metadata = {
                "rows": len(self.df),
                "columns": list(self.df.columns),
                "missing_values": self.df.isnull().sum().to_dict(),
                "dtypes": self.df.dtypes.apply(lambda x: str(x)).to_dict()
            }
            logger.log_step("DIAGNOSTICS", "Successfully loaded data with summary stats.", metadata)
            return self.df
        except Exception as e:
            logger.log_step("ERROR", f"Failed to load dataset: {str(e)}")
            raise

    def get_basic_stats(self):
        """Standard summary statistics."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        stats = self.df[numeric_cols].describe().to_dict()
        logger.log_step("SUMMARY_STATS", "Generated numeric summary stats.", {"stats": stats})
        return stats
