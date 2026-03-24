# AStats Prototype: A Workflow-First Reproducible Statistical Agent
![status](https://img.shields.io/badge/status-prototype-blue)

**A workflow-first agentic system that enforces statistical correctness and reproducibility before introducing LLM-driven flexibility.**

## Problem Statement
Current LLM-based statistical agents are frequently unreliable because they lack structured, well defined workflows. They often skip critical assumption checks, choose incorrect tests, or mis handle **repeated measures** data, leading to poor reproducibility and scientific errors.

## Approach: Agentic & Structure-Aware
1. **Auto-Discovery**: Automatic schema and diagnostic inference of any CSV.
2. **Structure-Aware Profiling**: Heuristics to detect grouping and **repeated measures** (Subject ID detection).
3. **Assumption-Driven Selection**: Systematic tests for normality (Shapiro-Wilk) and variance (Levene).
4. **LLM-assisted Explanation**: Uses a lightweight transformer-based model to generate natural-language justifications, while keeping statistical decision-making grounded in deterministic logic.
5. **Reproducible Logging**: Every decision is logged with specific justifications and metadata. **This ensures that every step in the analysis pipeline can be traced, audited, and reproduced.**

## 🔄 System Flow
1. **Load dataset** and perform initial diagnostics.
2. **Classify variables** (numeric / categorical).
3. **Detect structural patterns** (e.g., repeated measures).
4. **Validate statistical assumptions**.
5. **Select appropriate statistical test**.
6. **Execute test** and generate results.
7. **Log decisions** and generate explanations.

## Why This Matters
Statistical workflows require correctness, transparency, and reproducibility. By combining rule-based validation with agentic reasoning, this prototype demonstrates how LLMs can assist practitioners without compromising reliability.

## Design Insight
A key challenge in statistical automation is that errors often occur before test selection — at the stage of data structure interpretation (e.g., treating repeated measures as independent samples).

**This is particularly critical in real-world statistical practice, where incorrect structural assumptions can lead to invalid conclusions.**

This prototype explicitly addresses that by introducing a structure-aware profiling stage prior to statistical decision-making, ensuring that downstream test selection is grounded in correct assumptions about the data.

## Design Tradeoffs
- **Rule-based vs Fully Agentic Systems**: This system prioritizes rule-based statistical correctness before introducing agentic flexibility, to avoid unreliable or non-reproducible decisions.
- **Interpretability vs Automation**: Instead of fully automating decisions, the system logs every step to maintain transparency and auditability.
- **LLM Role**: LLMs are used for explanation rather than decision-making to ensure statistical validity is not compromised.

## Positioning
Traditional statistical tools (e.g., JASP, Jamovi) rely on predefined workflows and user-driven selection, while many LLM-based systems lack structure and reproducibility.

This prototype aims to bridge that gap by combining:
- **Structured statistical workflows** (like traditional tools)
- **Agentic assistance** (for flexibility and explanation)
- **Reproducible logging** (for transparency and auditability)

## 📁 Project Structure
- `main.py`: Pipeline orchestrator.
- `src/logger.py`: Decision-focused logging system for reproducibility.
- `src/eda.py`: Preliminary data loading and diagnostics.
- `src/profiler.py`: Variable type classification and **Structure Detection** (Repeated Measures).
- `src/test_selector.py`: Logic for selecting tests and checking assumptions.
- `src/executor.py`: Statistical test implementation (**t-test**, **Mann-Whitney**, with extensibility toward advanced models like Linear Mixed Models).
- `src/agent.py`: LLM-assisted explanation module.
- `data/sleepstudy.csv`: Primary dataset demonstrating structural awareness (repeated measures).

## Getting Started

### 1. Requirements
```bash
pip install pandas numpy scipy statsmodels transformers torch
```

### 2. Run Analysis
```bash
python3 main.py
```
A structured log is also saved as `workflow_log.json` for full reproducibility.


### Output Trace (Reproducible Decision Log)

```text
--- Starting Analysis for data/sleepstudy.csv ---
[LOAD] Loading dataset from data/sleepstudy.csv
[DIAGNOSTICS] Successfully loaded data with summary stats.
[CLASSIFY_VARIABLES] Classifying variables into numeric and categorical types.
--- DECISION: VARIABLE_CLASSIFICATION ---
Chosen: Inferred types (Numeric: Reaction, Categorical: Days)
Reason: Based on dtype and unique value heuristic.
------------------------------
[DETECT_STRUCTURE] Searching for repeated measures/ID structure.
[SYSTEM] Detected structure: Repeated Measures on Subject
[TEST_SELECTION] Selecting test for Reaction (numeric) vs Days (categorical)
[STRUCTURE_AWARE] Detected repeated measures via Subject. Checking for paired test.
--- DECISION: CHOOSE_TEST ---
Chosen: Friedman test
Reason: Repeated measures with >2 groups detected.
------------------------------
[EXECUTE_TEST] Running Friedman test
[RESULTS] Test Friedman test complete.
Final Decision: Friedman test - Results: {'statistic': 63.54, 'p_value': 2.8e-10}
```

---

## Future Work
- Extending structure detection to more complex experimental designs
- Integrating adaptive learning from user feedback
- Exploring hybrid rule-based + LLM-driven decision systems
- Scaling workflows to large datasets and distributed environments

## Conclusion
This prototype represents an initial step toward building reliable, workflow-first agentic systems for applied statistical practice, aligned with the goals of AStats.
