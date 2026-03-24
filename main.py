import os
import pandas as pd
from src.eda import EDA
from src.profiler import Profiler
from src.test_selector import TestSelector
from src.executor import Executor
from src.agent import AStatsAgent
from src.logger import logger

def run_agentic_pipeline(csv_path):
    print(f"\n--- Starting Analysis for {csv_path} ---")
    
    # 1. Initialize Agent
    agent = AStatsAgent() # Uses small LLM to justify choices
    
    # 2. EDA
    eda = EDA(csv_path)
    df = eda.load_data()
    eda.get_basic_stats()

    # 3. PROFILER (Auto-Discovery & Structure Detection)
    profiler = Profiler(df)
    vtypes = profiler.classify_variables()
    id_candidates = profiler.detect_repeated_measures()
    
    # Decide if we have a repeated measures structure
    id_col = None
    if id_candidates:
        id_col = id_candidates[0]['column']
        logger.log_step("SYSTEM", f"Detected structure: Repeated Measures on {id_col}")

    # 4. Agent Plans Analysis
    agent.plan_next_step("Profiling complete")

    # 5. Determine groups and numeric for comparison
    selector = TestSelector(df, vtypes)
    executor = Executor(df)

    # Heuristic for demo: find one categorical and one numeric
    num_cols = [c for c, t in vtypes.items() if t == 'numeric' and c != id_col]
    cat_cols = [c for c, t in vtypes.items() if t == 'categorical' and c != id_col]

    if num_cols and cat_cols:
        target_num = num_cols[0]
        target_cat = cat_cols[0]
        
        test_info = selector.select_test(target_num, target_cat, id_col=id_col)
        
        # 6. Agent Justifies choice
        agent.justify_test(test_info['test'], test_info.get('assumptions', "Checked structure"))
        
        # 7. Execute
        results = executor.run_test(test_info)
        print(f"Final Decision: {test_info['test']} - Results: {results}")

    # 8. Save session
    logger.save_log(f"workflow_{os.path.basename(csv_path)}.json")

if __name__ == "__main__":
    # Test on Normal/Non-Normal dataset
    run_agentic_pipeline("data/sample.csv")
    
    # Test on Sleepstudy (Repeated Measures)
    if os.path.exists("data/sleepstudy.csv"):
        run_agentic_pipeline("data/sleepstudy.csv")
