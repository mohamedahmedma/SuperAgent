"""The turn's rules: what a parent asked, which child, and whether the answer holds.

Not the LangGraph agent. That is a mechanism and it lives in `infrastructure/`; a contract
forbids langgraph and langchain from being imported here at all. The name is this product's
own vocabulary - SERVICES.md calls the service "chat agent + RAG", and `agent.tools` is a
profile field - so it is kept, and the precision is bought with the contract instead.

Expected residents (step 6F): turn_policy, clarification, answer_checks, answer_blocks,
finalize, model_output, child_names, child_resolution, and the pure half of rag - evidence,
confidence, policy, context_selection.
"""
