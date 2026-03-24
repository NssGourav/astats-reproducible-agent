try:
    import torch
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from src.logger import logger

class AStatsAgent:
    def __init__(self, model_name="google/flan-t5-small"):
        """Initialize the LLM-driven statistical agent."""
        logger.log_step("AGENT_INIT", f"Initializing AStats Agent with model: {model_name}")
        if not HAS_TRANSFORMERS:
            logger.log_step("AGENT_WARNING", "Transformers or Torch not found. Falling back to rule-based logic only.")
            self.generator = None
            self.is_connected = False
            return
            
        try:
            # We use a small model for memory efficiency and speed in the prototype
            self.generator = pipeline("text2text-generation", model=model_name, device=-1)
            self.is_connected = True
        except Exception as e:
            logger.log_step("AGENT_WARNING", f"Could not load LLM ({str(e)}). Falling back to rule-based logic only.")
            self.generator = None
            self.is_connected = False

    def think(self, context, question):
        """Standard agentic 'thought' process."""
        if not self.is_connected:
            return "Based on rule-based analysis, we proceed to the next step."

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        # Simple ReAct-style planning or justification
        output = self.generator(prompt, max_length=150)
        return output[0]['generated_text']

    def justify_test(self, test_name, assumptions):
        """Explain WHY a test was chosen in natural language."""
        context = f"The selected test is {test_name}. The assumptions checked are: {assumptions}."
        question = "Explain in one sentence why this test is appropriate for the given assumptions."
        
        reason = self.think(context, question)
        logger.log_decision(
            decision_type="LLM_JUSTIFICATION",
            chosen_action=test_name,
            reason=reason,
            metadata={"assumptions": assumptions}
        )
        return reason

    def plan_next_step(self, current_state):
        """Agent decides which module to call next."""
        context = f"Current workflow state: {current_state}. Available tools: [EDA, Profiler, TestSelector, Executor]."
        question = "Which tool should be used next to complete the statistical analysis? Respond with ONLY the tool name."
        
        plan = self.think(context, question)
        logger.log_step("AGENT_PLAN", f"Agent decided next step: {plan}")
        return plan
