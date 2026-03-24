import numpy as np
import scipy.stats as stats
import pandas as pd
from src.logger import logger

class Executor:
    def __init__(self, df):
        self.df = df

    def run_test(self, test_info):
        """Execute the chosen statistical test and return results."""
        test_name = test_info['test']
        logger.log_step("EXECUTE_TEST", f"Running {test_name}")

        num_col, cat_col = test_info.get('num_col'), test_info.get('cat_col')
        groups = test_info.get('groups')

        if test_name == 't-test':
            g1_data = self.df[self.df[cat_col] == groups[0]][num_col].dropna()
            g2_data = self.df[self.df[cat_col] == groups[1]][num_col].dropna()
            stat, p = stats.ttest_ind(g1_data, g2_data)
            results = {"statistic": float(stat), "p_value": float(p)}

        elif test_name == "Welch's t-test":
            g1_data = self.df[self.df[cat_col] == groups[0]][num_col].dropna()
            g2_data = self.df[self.df[cat_col] == groups[1]][num_col].dropna()
            stat, p = stats.ttest_ind(g1_data, g2_data, equal_var=False)
            results = {"statistic": float(stat), "p_value": float(p)}

        elif test_name == 'Mann-Whitney':
            g1_data = self.df[self.df[cat_col] == groups[0]][num_col].dropna()
            g2_data = self.df[self.df[cat_col] == groups[1]][num_col].dropna()
            stat, p = stats.mannwhitneyu(g1_data, g2_data)
            results = {"statistic": float(stat), "p_value": float(p)}

        elif test_name == 'paired_t_test':
            id_col = test_info['id_col']
            pivot = self.df.pivot(index=id_col, columns=cat_col, values=num_col)
            stat, p = stats.ttest_rel(pivot.iloc[:, 0], pivot.iloc[:, 1])
            results = {"statistic": float(stat), "p_value": float(p)}

        elif test_name == 'wilcoxon_signed_rank':
            id_col = test_info['id_col']
            pivot = self.df.pivot(index=id_col, columns=cat_col, values=num_col)
            stat, p = stats.wilcoxon(pivot.iloc[:, 0], pivot.iloc[:, 1])
            results = {"statistic": float(stat), "p_value": float(p)}

        elif test_name == 'Friedman test':
            id_col = test_info['id_col']
            pivot = self.df.pivot(index=id_col, columns=cat_col, values=num_col)
            # Friedman chisquare expect samples across columns
            stat, p = stats.friedmanchisquare(*[pivot[c] for c in pivot.columns])
            results = {"statistic": float(stat), "p_value": float(p)}

        elif test_name == 'chi_square':
            stat, p, dof, expected = stats.chi2_contingency(pd.crosstab(self.df[test_info['col1']], self.df[test_info['col2']]))
            results = {"statistic": float(stat), "p_value": float(p), "degrees_of_freedom": int(dof)}
            
        elif test_name == 'pearson_correlation':
            stat, p = stats.pearsonr(self.df[test_info['col1']], self.df[test_info['col2']])
            results = {"correlation": float(stat), "p_value": float(p)}
            
        else:
            results = {"error": f"Test {test_name} not implemented."}

        logger.log_step("RESULTS", f"Test {test_name} complete.", {"results": results})
        return results
