"""Every key the retrieval graph starts from is one its state declares.

LangGraph keeps only the keys named on the state schema and drops the rest without a
word. `RAGState` lost its `language` line in a refactor while `_initial_state` kept
writing it, so retrieval read None for the turn's language on every turn and the
Arabic and English halves of a paired document competed in every search. Nothing
failed; answers just got worse. This is the check that would have said so.
"""
import ast
import unittest
from pathlib import Path

from langgraph.graph import END, StateGraph

from backend.rag.graph_nodes import RAGState

PIPELINE = Path(__file__).resolve().parents[2] / "backend" / "rag" / "pipeline.py"


def _keys_initial_state_writes() -> list:
    """Read off the source: importing `backend.rag.pipeline` compiles the graph and
    reaches for the embedder, which this test has no business doing."""
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_initial_state"
    )
    returned = next(node for node in ast.walk(function) if isinstance(node, ast.Return))
    return [key.value for key in returned.value.keys]


class TheStateDeclaresEveryKeyTheGraphStartsFrom(unittest.TestCase):
    def test_no_initial_key_is_silently_dropped(self):
        undeclared = [key for key in _keys_initial_state_writes() if key not in RAGState.__annotations__]
        self.assertEqual(
            [], undeclared,
            "written by _initial_state and dropped by LangGraph: declare each on RAGState",
        )

    def test_the_language_reaches_a_node(self):
        """The regression itself, run through LangGraph rather than inferred from it."""
        seen = {}

        def probe(state):
            seen["language"] = state.get("language")
            seen["child_year"] = state.get("child_year")
            return {}

        graph = StateGraph(RAGState)
        graph.add_node("probe", probe)
        graph.set_entry_point("probe")
        graph.add_edge("probe", END)
        graph.compile().invoke({"question": "q", "language": "ar", "child_year": "Year 3", "sub_results": []})

        self.assertEqual({"language": "ar", "child_year": "Year 3"}, seen)


if __name__ == "__main__":
    unittest.main()
