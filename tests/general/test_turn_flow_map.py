"""The turn-flow diagram is a map of the code, and a map drifts quietly.

`backend/graphs/flow.py` draws one chat turn end to end. The compiled graphs in it — the
agent, the RAG graph, the ladder's rungs, the bound tools — are read from the code, but
the plain-Python steps around them are declared by name, and a renamed function would go
on being drawn as though it still ran. So the declared half is held to the code here:
every name must import, every note must belong to a node the graphs still have, and no
box may be left floating — a step with no edge is a step whose place in the turn was lost.

Nothing is rendered and nothing leaves the machine: the chart is built as text only.
"""
import unittest

from backend.graphs import flow


class TurnFlowMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flow = flow
        cls.parts = cls.flow.compile_parts()
        cls.chart, cls.compiled = cls.flow.build_flow(cls.parts)

    def test_every_declared_step_names_code_that_exists(self):
        self.assertEqual([], self.flow.stale_references())

    def test_every_edge_joins_two_drawn_steps(self):
        drawn = set(self.chart.nodes)
        dangling = [(a, b) for a, b, _, _ in self.chart.edges if a not in drawn or b not in drawn]
        self.assertEqual([], dangling)

    def test_no_step_is_left_unconnected(self):
        joined = {end for a, b, _, _ in self.chart.edges for end in (a, b)}
        self.assertEqual([], sorted(set(self.chart.nodes) - joined))

    def test_agent_notes_name_nodes_the_agent_still_has(self):
        stale = set(self.flow.AGENT_NOTES) | self.flow.AGENT_MODEL_NODES
        self.assertEqual(set(), stale - self.compiled["agent"])

    def test_rag_notes_name_nodes_one_of_its_shapes_still_has(self):
        """Either shape, because which one is live is a profile switch."""
        from backend.rag.pipeline import build_rag_graph

        nodes = set()
        for planning in (True, False):
            nodes |= {node.name for node in build_rag_graph(planning).get_graph().nodes.values()}
        named = set(self.flow.RAG_NOTES) | self.flow.RAG_MODEL_NODES | set(self.flow.RAG_RETRIEVAL_ENTRY)
        self.assertEqual(set(), named - nodes)

    def test_the_live_graphs_are_drawn_whole(self):
        """Every node the compiled graphs have appears in the chart — nothing is filtered out."""
        from backend.rag.pipeline import rag_graph

        rag = {n.name for n in rag_graph.get_graph().nodes.values()} - {"__start__", "__end__"}
        self.assertEqual(rag, self.compiled["rag"])
        self.assertIn("model", self.compiled["agent"])
        self.assertIn("tools", self.compiled["agent"])

    def test_the_target_flow_is_drawable_and_whole(self):
        """The proposal is held to shape only — never to the code, which is the point of it.

        It still has to be a graph: an edge naming a box nobody drew, or a box nothing
        reaches, is a hole in the design rather than a difference from today.
        """
        target = self.flow.build_target()
        drawn = set(target.nodes)
        self.assertEqual([], [(a, b) for a, b, _, _ in target.edges if a not in drawn or b not in drawn])
        joined = {end for a, b, _, _ in target.edges for end in (a, b)}
        self.assertEqual([], sorted(drawn - joined))
        self.assertIn("flowchart TD", target.render())

    def test_the_target_flow_is_not_confused_with_the_live_one(self):
        """Separate ids and separate lanes, so neither chart can quietly borrow the other."""
        target = self.flow.build_target()
        self.assertEqual(set(), set(target.nodes) & set(self.chart.nodes))
        self.assertEqual(set(), set(self.flow.TARGET_SECTIONS) & set(self.flow.SECTIONS))

    def test_every_bound_tool_is_drawn(self):
        labels = {label for label, _, _ in self.chart.nodes.values()}
        self.assertEqual([], [name for name in self.parts.tools if name not in labels])


if __name__ == "__main__":
    unittest.main()
