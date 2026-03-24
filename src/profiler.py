import pandas as pd
from src.logger import logger

class Profiler:
    def __init__(self, df):
        self.df = df
        self.variable_types = {}

    def classify_variables(self):
        """Categorize columns as 'numeric' or 'categorical' based on dtype and heuristics."""
        logger.log_step("CLASSIFY_VARIABLES", "Classifying variables into numeric and categorical types.")
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # If numeric but low unique values (< 5% of dataset and < 50), might be categorical
                unique_count = self.df[col].nunique()
                is_low_unique = (unique_count < 10) or (unique_count < 0.05 * len(self.df))
                
                if is_low_unique:
                    self.variable_types[col] = 'categorical'
                else:
                    self.variable_types[col] = 'numeric'
            else:
                self.variable_types[col] = 'categorical'
                
        logger.log_decision(
            decision_type="VARIABLE_CLASSIFICATION",
            chosen_action="Inferred types",
            reason="Based on dtype and unique value heuristic.",
            metadata=self.variable_types
        )
        return self.variable_types

    def detect_grouping_column(self):
        """Identify potential columns to group data for comparison (e.g., treatment groups)."""
        logger.log_step("DETECT_GROUPS", "Searching for suitable grouping columns for comparative analysis.")
        
        candidates = []
        for col, vtype in self.variable_types.items():
            if vtype == 'categorical':
                unique_vals = self.df[col].nunique()
                # A good group has at least 2 but not too many levels
                if 2 <= unique_vals <= 10:
                    candidates.append({
                        "column": col,
                        "unique_count": int(unique_vals),
                        "values": [str(v) for v in list(self.df[col].unique())]
                    })
        
        logger.log_decision(
            decision_type="GROUP_DETECTION",
            chosen_action="Identify candidate grouping columns",
            reason="Looking for low-cardinality categorical variables.",
            metadata={"candidates": candidates}
        )
        return candidates

    def detect_repeated_measures(self):
        """Identifies columns that might represent subjects or IDs repeated across groups."""
        logger.log_step("DETECT_STRUCTURE", "Searching for repeated measures/ID structure.")
        
        id_candidates = []
        for col in self.df.columns:
            # Heuristic: Name contains 'id', 'subject', 'sub', 'participant'
            name_lower = col.lower()
            if any(term in name_lower for term in ['id', 'subject', 'sub', 'participant']):
                # It's an ID candidate. Check if it repeats.
                counts = self.df[col].value_counts()
                is_repeated = (counts.max() > 1)
                
                id_candidates.append({
                    "column": col,
                    "is_repeated": is_repeated,
                    "max_repeats": int(counts.max()),
                    "unique_ids": int(self.df[col].nunique())
                })
        
        logger.log_decision(
            decision_type="STRUCTURE_DETECTION",
            chosen_action="Identify repeated measures",
            reason="Looking for ID-like column names with repeating entries.",
            metadata={"id_candidates": id_candidates}
        )
        return id_candidates
