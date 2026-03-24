from scipy import stats
from src.logger import logger

class TestSelector:
    def __init__(self, df, variable_types):
        self.df = df
        self.variable_types = variable_types

    def check_normality(self, data):
        """Perform Shapiro-Wilk test for normality."""
        if len(data) < 3:
            return False, 0 # Too few samples
        stat, p = stats.shapiro(data)
        return p > 0.05, p # Normal if p > 0.05

    def check_variance(self, group1, group2):
        """Perform Levene test for equal variance."""
        stat, p = stats.levene(group1, group2)
        return p > 0.05, p # Equal variance if p > 0.05

    def select_test(self, col1, col2, id_col=None):
        """Choose the most appropriate statistical test based on variable types and assumptions."""
        type1 = self.variable_types[col1]
        type2 = self.variable_types[col2]

        logger.log_step("TEST_SELECTION", f"Selecting test for {col1} ({type1}) vs {col2} ({type2})")

        # 1. Numeric vs Numeric
        if type1 == 'numeric' and type2 == 'numeric':
            return {
                "test": "pearson_correlation",
                "reason": "Both variables are numeric and we want to see their relationship.",
                "col1": col1, "col2": col2,
                "assumptions": {}
            }

        # 2. Categorical vs Categorical
        if type1 == 'categorical' and type2 == 'categorical':
            return {
                "test": "chi_square",
                "reason": "Both variables are categorical; checking for independence.",
                "col1": col1, "col2": col2,
                "assumptions": {}
            }

        # 3. Numeric vs Categorical (comparison)
        num_col, cat_col = (col1, col2) if type1 == 'numeric' else (col2, col1)
        
        # Check if categorical column has 2 groups (e.g. control/treatment)
        groups = self.df[cat_col].unique()
        
        if len(groups) == 2:
            g1_data = self.df[self.df[cat_col] == groups[0]][num_col].dropna()
            g2_data = self.df[self.df[cat_col] == groups[1]][num_col].dropna()

            # Handle Repeated Measures (Structure Awareness)
            if id_col:
                logger.log_step("STRUCTURE_AWARE", f"Detected repeated measures via {id_col}. Checking for paired test.")
                # We need to ensure we have same subjects in both groups for a paired test
                # This is a simplification for the prototype
                is_normal, _ = self.check_normality(g1_data - g2_data.values if len(g1_data) == len(g2_data) else g1_data)
                
                if is_normal:
                    test_name = "paired_t_test"
                    reason = "Repeated measures structure detected and differences are normal."
                else:
                    test_name = "wilcoxon_signed_rank"
                    reason = "Repeated measures structure detected and differences are non-normal."
                
                logger.log_decision("CHOOSE_TEST", test_name, reason, assumptions_checked={"normality_diff": is_normal})
                return {"test": test_name, "reason": reason, "num_col": num_col, "cat_col": cat_col, "groups": list(groups), "id_col": id_col}

            # Independent Groups
            is_normal1, p1 = self.check_normality(g1_data)
            is_normal2, p2 = self.check_normality(g2_data)
            is_normal = is_normal1 and is_normal2
            is_equal_var, pv = self.check_variance(g1_data, g2_data)

            assumptions = {
                "normality": is_normal,
                "homogeneity_of_variance": is_equal_var
            }

            if is_normal:
                test_name = "t-test"
                reason = "Groups are normal."
                if not is_equal_var:
                    test_name = "Welch's t-test"
                    reason += " Variance unequal; using Welch's."
            else:
                test_name = "Mann-Whitney"
                reason = "Groups are non-normal; using non-parametric."

            logger.log_decision("CHOOSE_TEST", test_name, reason, assumptions_checked=assumptions)
            return {"test": test_name, "reason": reason, "num_col": num_col, "cat_col": cat_col, "groups": list(groups)}
        
        # > 2 groups
        if len(groups) > 2:
            if id_col:
                test_name = "Friedman test"
                reason = "Repeated measures with >2 groups detected."
                return {"test": test_name, "reason": reason, "num_col": num_col, "cat_col": cat_col, "groups": list(groups), "id_col": id_col}
            else:
                test_name = "ANOVA"
                reason = "Independent groups >2; checking for ANOVA."
                return {"test": test_name, "reason": reason, "num_col": num_col, "cat_col": cat_col, "groups": list(groups)}

        return {"test": "unsupported", "reason": "No rule found."}
