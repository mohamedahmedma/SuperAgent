"""Guards for draw_rag_graph.py.

Two jobs. The first is that the generator produces syntactically valid Mermaid — a
broken diagram fails silently in the browser, long after anyone would connect it to
the change that broke it. The second is a drift guard: every node in the real
LangGraph must be named somewhere in the query diagram, so adding a graph node
without documenting it fails here rather than leaving the diagram quietly wrong.
"""
import importlib.util
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]

# Reserved words that must never be used as a node id.
MERMAID_KEYWORDS = {
    "graph", "subgraph", "end", "class", "classDef", "click", "style",
    "linkStyle", "direction", "flowchart", "default",
}


def load_module():
    """The generator lives at the repo root, not inside a package, so it is loaded by
    path. It must be registered in sys.modules BEFORE exec: @dataclass resolves its
    own module namespace through sys.modules and fails on an unregistered module."""
    if "draw_rag_graph" in sys.modules:
        return sys.modules["draw_rag_graph"]
    spec = importlib.util.spec_from_file_location("draw_rag_graph", REPO_ROOT / "draw_rag_graph.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DiagramGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()
        cls.graphs = {name: builder() for name, builder in cls.module.BUILDERS.items()}

    def test_all_views_are_registered(self):
        self.assertEqual({"ingest", "query", "delivery", "architecture"}, set(self.module.BUILDERS))

    def test_every_graph_validates(self):
        """validate() catches edges pointing at node ids that do not exist."""
        for name, graph in self.graphs.items():
            with self.subTest(view=name):
                graph.validate()

    def test_validate_rejects_a_dangling_edge(self):
        graph = self.module.Graph("t", "t", "t")
        graph.node("a", "A").edge("a", "ghost")
        with self.assertRaises(ValueError) as ctx:
            graph.validate()
        self.assertIn("ghost", str(ctx.exception))

    def test_validate_rejects_an_unknown_node_kind(self):
        graph = self.module.Graph("t", "t", "t")
        graph.node("a", "A", kind="sparkly")
        with self.assertRaises(ValueError):
            graph.validate()

    def test_validate_rejects_an_unknown_cluster(self):
        graph = self.module.Graph("t", "t", "t")
        graph.node("a", "A", cluster="nowhere")
        with self.assertRaises(ValueError):
            graph.validate()

    def test_mermaid_subgraphs_are_balanced(self):
        for name, graph in self.graphs.items():
            with self.subTest(view=name):
                source = self.module.render_mermaid(graph)
                opens = len(re.findall(r"^\s*subgraph ", source, re.M))
                ends = len(re.findall(r"^\s*end\s*$", source, re.M))
                self.assertEqual(opens, ends, f"{name}: {opens} subgraph vs {ends} end")

    def test_no_node_id_collides_with_a_mermaid_keyword(self):
        for name, graph in self.graphs.items():
            for node in graph.nodes:
                with self.subTest(view=name, node=node.id):
                    self.assertNotIn(node.id, MERMAID_KEYWORDS)

    def test_no_node_id_collides_with_a_style_class_name(self):
        """`class <ids> <name>` is ambiguous to read when an id equals a class name."""
        for name, graph in self.graphs.items():
            for node in graph.nodes:
                with self.subTest(view=name, node=node.id):
                    self.assertNotIn(node.id, self.module.STYLES)

    def test_node_ids_are_unique_within_a_graph(self):
        for name, graph in self.graphs.items():
            ids = [node.id for node in graph.nodes]
            with self.subTest(view=name):
                self.assertEqual(len(ids), len(set(ids)))

    def test_every_node_is_styled(self):
        for name, graph in self.graphs.items():
            source = self.module.render_mermaid(graph)
            classed = set()
            for group in re.findall(r"^\s*class ([^\s]+) \w+$", source, re.M):
                classed |= set(group.split(","))
            with self.subTest(view=name):
                self.assertEqual({node.id for node in graph.nodes}, classed)

    def test_labels_never_contain_an_unescaped_quote(self):
        """Mermaid labels are quoted; an inner quote ends the label early."""
        for name, graph in self.graphs.items():
            source = self.module.render_mermaid(graph)
            for line in source.splitlines():
                match = re.search(r'"(.*)"', line)
                if match:
                    with self.subTest(view=name, line=line[:60]):
                        self.assertNotIn('"', match.group(1))

    def test_every_declared_cluster_is_used(self):
        for name, graph in self.graphs.items():
            used = {node.cluster for node in graph.nodes if node.cluster}
            for cluster in graph.clusters:
                with self.subTest(view=name, cluster=cluster.id):
                    self.assertIn(cluster.id, used)


class DriftGuardTests(unittest.TestCase):
    """The diagram must keep describing the code it claims to describe."""

    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_every_langgraph_node_is_named_in_the_query_diagram(self):
        # The node names registered in backend/rag/pipeline.py build_rag_graph().
        langgraph_nodes = [
            "classify_complexity",
            "prepare_sub_questions",
            "retrieve_initial",
            "grade_documents",
            "rewrite_question",
            "retrieve_rewritten",
            "rag_sub_agent",
            "synthesis",
        ]
        source = self.module.render_mermaid(self.module.build_query_graph())
        for node_name in langgraph_nodes:
            with self.subTest(node=node_name):
                self.assertIn(node_name, source)

    def test_the_ingest_diagram_names_the_phase_3_stages(self):
        source = self.module.render_mermaid(self.module.build_ingest_graph())
        for stage in ["triage_image", "find_extraction", "save_extraction",
                      "render_surrogate", "asset_ids", "_hierarchy_chunks"]:
            with self.subTest(stage=stage):
                self.assertIn(stage, source)

    def test_the_delivery_diagram_names_the_phase_4_contract(self):
        source = self.module.render_mermaid(self.module.build_delivery_graph())
        for piece in ["AssetReference", "ClientCapabilities", "view_figure",
                      "record_observation", "/media/resolve", "inline_data"]:
            with self.subTest(piece=piece):
                self.assertIn(piece, source)

    def test_the_architecture_diagram_names_every_profile_section(self):
        from backend.profiles.schema import DomainProfile

        source = self.module.render_mermaid(self.module.build_architecture_graph())
        sections = set(DomainProfile.model_fields) - {"profile_version", "name", "extends"}
        for section in sections:
            with self.subTest(section=section):
                self.assertIn(section, source)


class OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_a_full_run_writes_every_artifact(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            exit_code = self.module.main(["--out", str(out), "--no-png"])
            self.assertEqual(0, exit_code)
            for view in ("ingest", "query", "delivery", "architecture"):
                self.assertTrue((out / f"{view}.mmd").is_file())
                self.assertTrue((out / f"{view}.md").is_file())
            self.assertTrue((out / "index.html").is_file())

    def test_a_single_view_can_be_rendered_alone(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.module.main(["--view", "ingest", "--out", str(out), "--no-png"])
            self.assertTrue((out / "ingest.mmd").is_file())
            self.assertFalse((out / "query.mmd").exists())

    def test_the_markdown_wraps_the_diagram_in_a_mermaid_fence(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.module.main(["--view", "query", "--out", str(out), "--no-png"])
            text = (out / "query.md").read_text(encoding="utf-8")
            self.assertIn("```mermaid", text)
            self.assertIn("flowchart TB", text)
            self.assertEqual(2, text.count("```"))

    def test_the_html_viewer_embeds_every_diagram(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.module.main(["--out", str(out), "--no-png"])
            page = (out / "index.html").read_text(encoding="utf-8")
            self.assertEqual(4, page.count('<pre class="mermaid">'))
            self.assertIn("mermaid.esm.min.mjs", page)

    def test_missing_graphviz_is_reported_not_raised(self):
        """The dot binary is an OS-level dependency; its absence must not fail a run."""
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            with patch.object(self.module.shutil, "which", return_value=None):
                paths, reason = self.module.write_pngs(
                    [self.module.build_ingest_graph()], Path(tmp)
                )
        self.assertEqual([], paths)
        self.assertIn("dot", reason)


if __name__ == "__main__":
    unittest.main()
