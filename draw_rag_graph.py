"""Render the system's pipelines as diagrams.

Four views, because the system is no longer one graph:

  1. ingest       — upload -> parse -> figure enrichment -> chunk hierarchy -> stores
  2. query        — the LangGraph RAG flow, plus the sub-steps hidden inside its nodes
  3. delivery     — how a retrieved image reaches any client, UI-agnostically
  4. architecture — which layer owns what, and where the DomainProfile reaches

Usage:
    uv run python draw_rag_graph.py                # all four, to ./diagrams
    uv run python draw_rag_graph.py --view ingest  # just one
    uv run python draw_rag_graph.py --out build/   # elsewhere

Every run writes Mermaid (`.mmd`), a Markdown file you can preview directly in VS Code,
and a standalone HTML viewer. PNGs are written too, but only when the Graphviz `dot`
binary is on PATH — it is an OS-level dependency, so it is optional rather than
required, and its absence is reported instead of crashing the run.

Each graph is defined ONCE as data (`Graph`) and rendered by two backends. Adding a
node means editing one builder, not keeping two dialects in sync.
"""

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Graph model
# ---------------------------------------------------------------------------

# kind -> (mermaid shape open/close, fill, stroke). One vocabulary, both renderers.
STYLES: Dict[str, Tuple[str, str, str, str]] = {
    "entry":     ("([", "])", "#E8E8E8", "#4A4A4A"),
    "step":      ("[", "]", "#DCE9F8", "#2C5F9E"),
    "llm":       ("[", "]", "#FDE9D9", "#C0722C"),
    "decision":  ("{{", "}}", "#FCF3CF", "#9A7D0A"),
    "store":     ("[(", ")]", "#E6E0F8", "#5B4B9E"),
    "asset":     ("[", "]", "#DFF3EC", "#2E7D63"),
    "ok":        ("[", "]", "#DDF2DD", "#3A7D3A"),
    "hitl":      ("[", "]", "#EBDDF5", "#6C3A9E"),
    "none":      ("[", "]", "#F5DDDD", "#9E3A3A"),
    "profile":   ("[", "]", "#FFF3CD", "#B8860B"),
}

GRAPHVIZ_SHAPES = {
    "entry": "oval", "decision": "diamond", "store": "cylinder",
}


@dataclass
class Node:
    id: str
    label: str
    kind: str = "step"
    cluster: Optional[str] = None


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    dashed: bool = False


@dataclass
class Cluster:
    id: str
    label: str


@dataclass
class Graph:
    name: str
    title: str
    description: str
    direction: str = "TB"
    clusters: List[Cluster] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)

    def cluster(self, cluster_id: str, label: str) -> "Graph":
        self.clusters.append(Cluster(cluster_id, label))
        return self

    def node(self, node_id: str, label: str, kind: str = "step", cluster: Optional[str] = None) -> "Graph":
        self.nodes.append(Node(node_id, label, kind, cluster))
        return self

    def edge(self, src: str, dst: str, label: str = "", dashed: bool = False) -> "Graph":
        self.edges.append(Edge(src, dst, label, dashed))
        return self

    def validate(self) -> None:
        """Catch a typo'd node id here rather than as a mystery node in the output."""
        known = {node.id for node in self.nodes}
        cluster_ids = {cluster.id for cluster in self.clusters}
        problems = []
        for edge in self.edges:
            for endpoint in (edge.src, edge.dst):
                if endpoint not in known:
                    problems.append(f"edge references unknown node {endpoint!r}")
        for node in self.nodes:
            if node.cluster and node.cluster not in cluster_ids:
                problems.append(f"node {node.id!r} references unknown cluster {node.cluster!r}")
            if node.kind not in STYLES:
                problems.append(f"node {node.id!r} has unknown kind {node.kind!r}")
        if problems:
            raise ValueError(f"{self.name}: " + "; ".join(sorted(set(problems))))


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _mermaid_label(label: str) -> str:
    """Mermaid labels are quoted, so the quote character itself must go, and newlines
    become <br/>."""
    return label.replace('"', "'").replace("\n", "<br/>")


def render_mermaid(graph: Graph) -> str:
    graph.validate()
    lines = [f"flowchart {graph.direction}"]

    by_cluster: Dict[Optional[str], List[Node]] = {}
    for node in graph.nodes:
        by_cluster.setdefault(node.cluster, []).append(node)

    def emit(node: Node, indent: str) -> None:
        open_shape, close_shape, _, _ = STYLES[node.kind]
        lines.append(f'{indent}{node.id}{open_shape}"{_mermaid_label(node.label)}"{close_shape}')

    for node in by_cluster.get(None, []):
        emit(node, "    ")

    for cluster in graph.clusters:
        members = by_cluster.get(cluster.id, [])
        if not members:
            continue
        lines.append(f'    subgraph {cluster.id}["{_mermaid_label(cluster.label)}"]')
        lines.append("        direction TB")
        for node in members:
            emit(node, "        ")
        lines.append("    end")

    for edge in graph.edges:
        arrow = "-.->" if edge.dashed else "-->"
        if edge.label:
            lines.append(f'    {edge.src} {arrow}|"{_mermaid_label(edge.label)}"| {edge.dst}')
        else:
            lines.append(f"    {edge.src} {arrow} {edge.dst}")

    used_kinds = {node.kind for node in graph.nodes}
    for kind in sorted(used_kinds):
        _, _, fill, stroke = STYLES[kind]
        lines.append(f"    classDef {kind} fill:{fill},stroke:{stroke},stroke-width:1px,color:#111")
    for kind in sorted(used_kinds):
        members = ",".join(node.id for node in graph.nodes if node.kind == kind)
        lines.append(f"    class {members} {kind}")

    return "\n".join(lines)


# Fixed-width tag per kind, for the text renderer. ASCII on purpose: this is printed
# to a terminal, and a Windows console in a non-UTF-8 codepage turns box-drawing
# characters into mojibake — which is exactly where someone reads this from.
TEXT_TAGS = {
    "entry": "ENTRY ", "step": "STEP  ", "llm": "MODEL ", "decision": "DECIDE",
    "store": "STORE ", "asset": "ASSET ", "ok": "OK    ", "hitl": "ASK   ",
    "none": "STOP  ", "profile": "CONF  ",
}


def render_text(graph: Graph, width: int = 96) -> str:
    """The graph as printable text: nodes grouped by stage, then every edge.

    Exists because the Mermaid and PNG outputs need a viewer, and the question this
    answers most often — "what actually runs, in what order, and which steps cost a
    model call" — is asked at a terminal. MODEL-tagged nodes are the priced ones, so a
    reader can count a turn's calls straight off the node list.
    """
    rule = "=" * width
    lines = [rule, graph.title, rule, "", graph.description, ""]

    by_cluster: Dict[Optional[str], List[Node]] = {}
    for node in graph.nodes:
        by_cluster.setdefault(node.cluster, []).append(node)

    def emit(node: Node, indent: str) -> None:
        head, *rest = node.label.split("\n")
        lines.append(f"{indent}[{TEXT_TAGS[node.kind]}] {node.id:<18} {head}")
        for extra in rest:
            lines.append(f"{indent}{' ' * (len(TEXT_TAGS[node.kind]) + 3)} {'':<18} {extra}")

    lines.append("NODES")
    lines.append("-" * width)
    loose = by_cluster.get(None, [])
    if loose:
        lines.append("  (top level)")
        for node in loose:
            emit(node, "    ")
        lines.append("")
    for cluster in graph.clusters:
        members = by_cluster.get(cluster.id, [])
        if not members:
            continue
        lines.append(f"  {cluster.label}")
        for node in members:
            emit(node, "    ")
        lines.append("")

    lines.append("FLOW")
    lines.append("-" * width)
    for edge in graph.edges:
        arrow = "..>" if edge.dashed else "-->"
        label = edge.label.replace("\n", " ")
        suffix = f"   # {label}" if label else ""
        lines.append(f"  {edge.src:<20} {arrow} {edge.dst:<20}{suffix}")

    priced = [node.id for node in graph.nodes if node.kind == "llm"]
    lines += ["", "-" * width, f"MODEL calls on this graph ({len(priced)}): " + ", ".join(priced), ""]
    return "\n".join(lines)


def render_graphviz(graph: Graph):
    """Graphviz Digraph, or None when the `graphviz` package is absent."""
    try:
        import graphviz
    except ImportError:
        return None

    graph.validate()
    dot = graphviz.Digraph(graph.name, format="png")
    dot.attr(rankdir=graph.direction, splines="spline", fontname="Helvetica",
             nodesep="0.4", ranksep="0.6", bgcolor="white", label=graph.title,
             labelloc="t", fontsize="18")
    dot.attr("node", fontname="Helvetica", fontsize="10")
    dot.attr("edge", fontname="Helvetica", fontsize="9", color="#555555")

    def attrs(kind: str) -> dict:
        _, _, fill, stroke = STYLES[kind]
        shape = GRAPHVIZ_SHAPES.get(kind, "box")
        style = "filled" if shape in ("diamond", "oval", "cylinder") else "rounded,filled"
        return {"shape": shape, "style": style, "fillcolor": fill, "color": stroke}

    by_cluster: Dict[Optional[str], List[Node]] = {}
    for node in graph.nodes:
        by_cluster.setdefault(node.cluster, []).append(node)

    for node in by_cluster.get(None, []):
        dot.node(node.id, node.label, **attrs(node.kind))

    for cluster in graph.clusters:
        members = by_cluster.get(cluster.id, [])
        if not members:
            continue
        with dot.subgraph(name=f"cluster_{cluster.id}") as sub:
            sub.attr(label=cluster.label, style="filled", fillcolor="#FBFBFB",
                     color="#999999", fontsize="13", fontname="Helvetica-Bold")
            for node in members:
                sub.node(node.id, node.label, **attrs(node.kind))

    for edge in graph.edges:
        dot.edge(edge.src, edge.dst, label=edge.label, style="dashed" if edge.dashed else "solid")
    return dot


# ---------------------------------------------------------------------------
# 1. Ingest pipeline
# ---------------------------------------------------------------------------

def build_ingest_graph() -> Graph:
    g = Graph(
        "ingest", "Ingest pipeline — document to indexed chunks",
        "How an uploaded file becomes retrievable text, including the figure path "
        "added in Phase 3.",
    )
    g.cluster("parse", "Layout parsing — backend/indexing/*_layout.py")
    g.cluster("enrich", "Figure enrichment — backend/assets/ (Phase 3)")
    g.cluster("chunk", "Chunk hierarchy — backend/indexing/document_loader.py")
    g.cluster("persist", "Persistence")

    g.node("upload", "POST /documents/upload\nupload_document_async", "entry")
    g.node("supported", "is_supported_document\nprofile.ingest.supported_extensions", "decision")
    g.node("reject", "400 — profile.user_copy.unsupported_file_type", "none")
    g.node("cleanup", "delete_document_transactionally\nMilvus + ParentChunk + Redis\n+ DocumentAsset + orphan blobs", "step")

    g.node("parse_blocks", "parse_pdf / docx / html / xlsx _blocks\nheading | text | table | IMAGE", "step", "parse")
    g.node("furniture", "remove_page_furniture\nrepeated header/footer lines", "step", "parse")

    g.node("has_images", "any image blocks?", "decision", "enrich")
    g.node("triage", "triage_image\nsize / aspect / bytes / page-furniture\n-> role + tier + reason", "decision", "enrich")
    g.node("dropped", "SKIPPED\nrecorded with reason, no blob stored", "none", "enrich")
    g.node("blob", "BlobStore.put(sha256)\ncontent-addressed, idempotent", "store", "enrich")
    g.node("cache", "AssetStore.find_extraction\nkey = sha256 + profile + version", "decision", "enrich")
    g.node("extract", "FigureExtractor\nVision (structured) or Heuristic (alt/caption)\nfallback on failure", "llm", "enrich")
    g.node("save_extract", "AssetStore.save_extraction\nglobal cache — paid once per distinct image", "asset", "enrich")
    g.node("dossier", "AssetDossier recorded\nDocumentAsset row per occurrence", "asset", "enrich")
    g.node("surrogate", "render_surrogate()\ncaption + description + transcription + tags\n-> text block carrying asset_ids", "asset", "enrich")

    g.node("stitch", "_stitch_cross_page_blocks\nrejoin paragraphs/tables split by a page break", "step", "chunk")
    g.node("units", "_blocks_to_units\nsection stack -> section path per unit\nkind = text | table | figure", "step", "chunk")
    g.node("hierarchy", "_hierarchy_chunks\nL1 -> L2 -> L3 packing\nfigures & tables isolated at leaf\nasset_ids + modality propagated", "step", "chunk")

    g.node("split", "chunk_level", "decision")
    g.node("parents", "ParentChunkStore.upsert\nL1/L2 -> Postgres + Redis\n(text, modality, asset_ids)", "store", "persist")
    g.node("dedup", "MilvusWriter\nexact-hash dedup before embedding", "step", "persist")
    g.node("embed", "EmbeddingService\nbge-m3 dense vectors", "llm", "persist")
    g.node("milvus", "Milvus collection\ndense + BM25 sparse\n+ modality + asset_ids", "store", "persist")

    g.edge("upload", "supported")
    g.edge("supported", "reject", "unsupported", dashed=True)
    g.edge("supported", "cleanup", "supported")
    g.edge("cleanup", "parse_blocks")
    g.edge("parse_blocks", "furniture")
    g.edge("furniture", "has_images")
    g.edge("has_images", "stitch", "no images", dashed=True)
    g.edge("has_images", "triage", "images found")
    g.edge("triage", "dropped", "drop")
    g.edge("triage", "blob", "figure")
    g.edge("blob", "cache")
    g.edge("cache", "dossier", "hit — no model call", dashed=True)
    g.edge("cache", "extract", "miss")
    g.edge("extract", "save_extract")
    g.edge("save_extract", "dossier")
    g.edge("dossier", "surrogate")
    g.edge("surrogate", "stitch")
    g.edge("stitch", "units")
    g.edge("units", "hierarchy")
    g.edge("hierarchy", "split")
    g.edge("split", "parents", "L1 / L2")
    g.edge("split", "dedup", "L3 (leaf)")
    g.edge("dedup", "embed")
    g.edge("embed", "milvus")
    return g


# ---------------------------------------------------------------------------
# 2. Query pipeline
# ---------------------------------------------------------------------------

def _retrieval_cluster(g: Graph, suffix: str, title: str) -> str:
    """Recall -> auto-merge -> rerank -> threshold. Reused by every retrieval site."""
    cluster_id = f"retrieval_{suffix}"
    g.cluster(cluster_id, title)
    g.node(f"recall_{suffix}", "Milvus hybrid search\ndense + BM25 sparse, RRF fused\nfilter chunk_level == LEAF_RETRIEVE_LEVEL\ncandidate_k from profile or env", "step", cluster_id)
    g.node(f"merge_{suffix}", "Auto-merge L3 -> L2 -> L1\ngroup by parent_chunk_id\nparent text from Postgres DocStore", "step", cluster_id)
    g.node(f"rerank_{suffix}", "Rerank (Cohere-compatible)\nskipped when RERANK_* unset", "step", cluster_id)
    g.node(f"filter_{suffix}", "Threshold filter\nscore >= RERANK_MIN_SCORE", "step", cluster_id)
    g.edge(f"recall_{suffix}", f"merge_{suffix}")
    g.edge(f"merge_{suffix}", f"rerank_{suffix}")
    g.edge(f"rerank_{suffix}", f"filter_{suffix}")
    return f"filter_{suffix}"


def build_query_graph() -> Graph:
    g = Graph(
        "query", "Query pipeline — turn planning, then corrective RAG",
        "Everything a message passes through: the pre-agent turn planner (which can end "
        "a turn before any agent exists), then the LangGraph RAG nodes and the sub-steps "
        "hidden inside them.",
    )
    # Stage 0 is NOT a LangGraph node and deliberately so — see the module docstring of
    # backend/chat/orchestrator.py. It runs before the agent is constructed, which is
    # the only place a turn can still be made cheaper.
    g.cluster("entrycluster", "0. Turn entry — backend/chat/service.py::_enter_turn")
    g.cluster("resolution", "1. Contextual resolution — backend/chat/resolution.py")
    g.cluster("signals", "2. Signal ladder — backend/chat/signals.py (cheapest rung first)")
    g.cluster("policy", "3. Turn policy — backend/chat/turn_policy.py (pure)")
    g.cluster("plan", "4. Complexity planning — classify_complexity (profile.rag.complexity_planning_enabled)")
    g.cluster("subagents", "6. Parallel sub-agents — rag_sub_agent x N via Send() (planning only)")
    g.cluster("grade", "8. Evidence assessment — grade_documents_node")
    g.cluster("ladder", "The assessor ladder — cheapest rung first, stops at profile.rag.evidence_required_certainty")
    g.cluster("rewrite", "9. Query rewrite — rewrite_question -> retrieve_rewritten, max profile.rag.max_rewrites")
    g.cluster("resume", "HITL resume — resume_rag_from_hitl (separate entry point)")

    g.node("inbound", "User message + session history\nchat_with_agent_stream", "entry")

    # --- 0. turn entry -----------------------------------------------------------
    g.node("pending", "pending clarification in session metadata?", "decision", "entrycluster")
    g.node("hitl_resolve", "resolve the reply against the conversation\nAND the question that was asked\nchat/resolve_question.j2 + hitl options", "llm", "entrycluster")
    g.node("supersede", "correction / new_topic?\nResolvedQuestion.supersedes_pending_question", "decision", "entrycluster")

    # --- 1. resolution -----------------------------------------------------------
    g.node("resolve_gate", "needs_resolution — local, no model\nfirst message? marker? opener?\nshorter than query_resolution_max_chars?", "decision", "resolution")
    g.node("resolver", "FAST_MODEL structured call\n-> question + constraints + intent\nabstains to the raw message on any failure", "llm", "resolution")
    g.node("resolved", "ResolvedQuestion\nthe turn's ONE subject: scoring, retrieval,\nthe agent prompt and the resume path all read this", "step", "resolution")

    # --- 2. signal ladder --------------------------------------------------------
    g.node("social", "SocialDetector — whole-message lookup\nprofile.agent.social_phrases\ncertainty HIGH, no model", "decision", "signals")
    g.node("scope_cat", "CatalogueScopeDetector\nembed(text_to_score) vs question catalogue\nscore vs the corpus's DERIVED floor\nmay ADMIT, may never refuse", "decision", "signals")
    g.node("directions", "distinct_directions\nat/above floor + paraphrases collapsed\n+ trailing options dropped", "step", "signals")
    g.node("scope_model", "ScopeModelDetector — FAST_MODEL\nshown rung 1's matches and the floor\nthe ONLY rung that may end a turn", "llm", "signals")

    # --- 3. turn policy ----------------------------------------------------------
    g.node("turnplan", "resolve_turn -> TurnPlan\nstatic_reply · exposed_tools · retrieval_sections\nscope_options · resolved_question · carried_constraints", "decision", "policy")
    g.node("static_ood", "Out of domain, confirmed by a rung that READ it\nprofile.user_copy.out_of_domain\nno agent, no prompt, no tool schema", "none", "policy")
    g.node("social_reply", "Social turn — every tool unbound\none lean model call, or static copy", "ok", "policy")
    g.node("agentbuild", "create_agent_for_request\n+ agent/turn_context.j2 (resolved question\nand carried conditions, only when there are any)", "step", "policy")

    g.node("start", "search_knowledge_base(query)\n-> run_rag_graph", "entry")
    g.node("constrain", "_search_query = apply_constraints\nappends conditions the agent's query omitted\nadditive — never a filter", "step")

    g.node("fastpath", "Local rules — no LLM\n_simple_question_fast_path_reason\nprofile.rag.*_query_markers", "decision", "plan")
    g.node("planner", "FAST_MODEL structured call\n-> ComplexityResult\nsub_questions capped by profile", "llm", "plan")
    g.node("complexity", "simple or complex?", "decision", "plan")

    g.node("retrieve_initial", "retrieve_initial", "step")
    filt_initial = _retrieval_cluster(g, "initial", "2. Retrieval pipeline (initial)")

    g.node("prepare_subs", "prepare_sub_questions", "step")
    g.node("fanout", "Send() fan-out\none sub-agent per sub-question", "step")
    g.node("sub_retrieve", "retrieve_initial (per sub-question)", "step", "subagents")
    g.node("sub_grade", "grade_documents (no rewrite pass)", "step", "subagents")
    g.node("synthesis", "4. Synthesis\ndedupe_documents by chunk_id\nmerge sub_traces, aggregate route\noutage outranks HITL", "step")

    g.node("rung0", "Rung 0 — structural\nno documents is conclusive\ncertainty HIGH", "decision", "ladder")
    g.node("gradeq", "graded against the CONSTRAINED question\nevidence covering every year group does not\nsettle a question about Year 6", "step", "grade")
    g.node("rung1", "Rung 1 — lexical overlap\ncertainty LOW: reports, never concludes", "decision", "ladder")
    g.node("rung2", "Rung 2 — cross-encoder\ncalibrated per-chunk relevance\ncertainty MEDIUM", "decision", "ladder")
    g.node("rung3", "Rung 3 — GRADE_MODEL structured call\n-> EvidenceGrade + supporting_chunks\ncertainty HIGH", "llm", "ladder")
    g.node("report", "EvidenceReport\nper-chunk scores, certainty, provenance\nthe single source of truth", "step", "grade")
    g.node("route", "decide_route — pure policy\nrefuses to act below the required certainty\nis_followup suppresses catalogued directions:\na subject settled last turn is not a choice to offer", "decision", "grade")
    g.node("ctxpolicy", "select_context_indices — pure policy\ntrims only to chunks an assessment named\nrequires MEDIUM certainty", "decision", "grade")

    g.node("budget", "rewrite budget left?", "decision", "rewrite")
    g.node("rewriter", "FAST_MODEL structured call\n-> RewritePlan (exactly one method)", "llm", "rewrite")
    g.node("stepback", "Step-back\nabstract the question", "step", "rewrite")
    g.node("hyde", "HyDE\nhypothetical answer document\nretrieval aid only, never evidence", "step", "rewrite")
    filt_rewritten = _retrieval_cluster(g, "rewritten", "Retrieval pipeline (rewritten query)")

    g.node("answer", "Answer generation\nchunks formatted with [n] citations\nknowledge_result.j2 states the carried conditions,\nso they bind the ANSWER and not only the search", "ok")
    g.node("hitl_pause", "HITL pause\npending_hitl saved to the session\ncarried_constraints ride the resume state", "hitl")
    g.node("nokb", "No usable evidence\nprofile.user_copy.no_knowledge", "none")
    g.node("outage", "Retrieval error\nstatic retry notice — never no_knowledge", "none")

    g.node("hitl_answer", "resume_rag_from_hitl\nreceives the resolved reply, the USER's original\nquestion, and the conversation", "entry", "resume")
    g.node("hitl_refine", "_refined_question_for_hitl\nthe resolved question replaces the old reading\n(concatenation could only ever add to it)", "step", "resume")
    g.node("hitl_targeted", "Targeted retrieval\nskips planning and decomposition", "step", "resume")
    filt_hitl = _retrieval_cluster(g, "hitl", "Retrieval pipeline (HITL targeted)")

    # --- edges: turn entry -------------------------------------------------------
    g.edge("inbound", "pending")
    g.edge("pending", "hitl_resolve", "yes")
    g.edge("pending", "resolve_gate", "no — ordinary turn")
    g.edge("hitl_resolve", "supersede")
    g.edge("supersede", "hitl_answer", "no — the reply answers it")
    g.edge("supersede", "resolved", "yes — abandon it, run a fresh turn", dashed=True)

    # --- edges: resolution -------------------------------------------------------
    g.edge("resolve_gate", "resolver", "referential or short")
    g.edge("resolve_gate", "resolved", "stands on its own — no call", dashed=True)
    g.edge("resolver", "resolved")
    g.edge("resolved", "social")

    # --- edges: signal ladder ----------------------------------------------------
    g.edge("social", "social_reply", "whole message is a pleasantry", dashed=True)
    g.edge("social", "scope_cat", "a real message")
    g.edge("scope_cat", "directions")
    g.edge("directions", "turnplan", "at or above floor — ADMIT, no model", dashed=True)
    g.edge("directions", "scope_model", "below floor — escalate")
    g.edge("scope_model", "turnplan")

    # --- edges: turn policy ------------------------------------------------------
    g.edge("turnplan", "static_ood", "out_of_domain at HIGH certainty", dashed=True)
    g.edge("turnplan", "agentbuild", "anything less — tool stays bound AND working")
    g.edge("agentbuild", "start")
    g.edge("start", "constrain")
    g.edge("constrain", "fastpath")
    g.edge("fastpath", "complexity", "confidently simple")
    g.edge("fastpath", "planner", "not confident")
    g.edge("planner", "complexity")
    g.edge("complexity", "retrieve_initial", "simple")
    g.edge("complexity", "prepare_subs", "complex")
    g.edge("retrieve_initial", "recall_initial")
    g.edge(filt_initial, "gradeq")
    g.edge("gradeq", "rung0")
    g.edge("prepare_subs", "fanout")
    g.edge("fanout", "sub_retrieve")
    g.edge("sub_grade", "synthesis")
    g.edge("synthesis", "answer", "docs merged")
    g.edge("synthesis", "hitl_pause", "clarify / scope_select", dashed=True)
    g.edge("synthesis", "nokb", "no evidence", dashed=True)
    g.edge("synthesis", "outage", "sub-agent outage", dashed=True)
    g.edge("rung0", "rung1", "documents exist")
    g.edge("rung0", "report", "none -> conclusive, no LLM", dashed=True)
    g.edge("rung1", "rung2", "cannot conclude on word overlap")
    g.edge("rung2", "report", "clearly sufficient or clearly irrelevant", dashed=True)
    g.edge("rung2", "rung3", "inconclusive middle band")
    g.edge("rung3", "report")
    g.edge("report", "route")
    g.edge("route", "ctxpolicy", "answer")
    g.edge("ctxpolicy", "answer", "sends the chunks that carried the evidence")
    g.edge("route", "hitl_pause", "clarify / scope_select\ndirections are asked BEFORE a rewrite is spent —\nbut never on a follow-up, and never as paraphrases", dashed=True)
    g.edge("route", "nokb", "no_knowledge — nothing retrieved, or off-subject", dashed=True)
    g.edge("route", "outage", "retrieval_error", dashed=True)
    g.edge("route", "budget", "rewrite")
    g.edge("budget", "answer", "spent or unplannable — answer from what the passes found", dashed=True)
    g.edge("budget", "rewriter", "available")
    g.edge("rewriter", "stepback")
    g.edge("rewriter", "hyde")
    g.edge("stepback", "recall_rewritten")
    g.edge("hyde", "recall_rewritten")
    g.edge(filt_rewritten, "rung0", "re-assess the union with the first pass", dashed=True)
    g.edge("hitl_answer", "hitl_refine")
    g.edge("hitl_refine", "hitl_targeted")
    g.edge("hitl_targeted", "recall_hitl")
    g.edge(filt_hitl, "gradeq", "re-assess", dashed=True)
    return g


# ---------------------------------------------------------------------------
# 3. Architecture
# ---------------------------------------------------------------------------

def build_architecture_graph() -> Graph:
    g = Graph(
        "architecture", "System architecture — layers, stores, and profile reach",
        "What the DomainProfile controls, and which store owns which artifact.",
        direction="LR",
    )
    g.cluster("cfg", "Configuration — backend/profiles/")
    g.cluster("api", "API — backend/api/")
    g.cluster("agentlayer", "Agent & RAG — backend/chat/, backend/rag/")
    g.cluster("ingestlayer", "Ingest — backend/indexing/, backend/assets/")
    g.cluster("stores", "Stores")

    # Every section listed here is asserted against DomainProfile.model_fields by
    # tests/test_diagrams.py, so a new profile section cannot ship undocumented.
    g.node("domain_profile", "DomainProfile (YAML, composable)\nidentity · models · agent · rag\nretrieval · chunking · assets\nuser_copy · ingest", "profile", "cfg")
    g.node("envover", "Environment overrides\nenv > profile > schema default", "profile", "cfg")

    g.node("routes", "FastAPI routes\nchat · documents · sessions · auth", "step", "api")
    g.node("agent", "LangGraph agent\ntools from profile.agent.tools\nprompt from profile persona", "step", "agentlayer")
    g.node("ragflow", "RAG graph\nplan -> retrieve -> grade -> rewrite", "step", "agentlayer")

    g.node("loader", "DocumentLoader\nlayout parse -> enrich -> L1/L2/L3", "step", "ingestlayer")
    g.node("figures", "FigurePipeline\ntriage -> cache -> extract", "asset", "ingestlayer")
    g.node("delivery", "AssetPresenter + /media routes\ncapability-driven renditions\nno UI knowledge on this side", "asset", "agentlayer")

    g.node("milvus", "Milvus\nleaf chunks, dense + BM25\nmodality, asset_ids", "store", "stores")
    g.node("postgres", "PostgreSQL\nParentChunk · DocumentAsset\nAssetExtraction · chat history", "store", "stores")
    g.node("redis", "Redis\nhot parent chunks & dossiers\nprofile-scoped key prefix", "store", "stores")
    g.node("blobs", "Blob store\ncontent-addressed images\nlocal filesystem or S3/MinIO", "store", "stores")

    g.edge("envover", "domain_profile", "overlaid at load")
    g.edge("domain_profile", "routes", "api_title, extensions")
    g.edge("domain_profile", "agent", "persona, tools, budgets")
    g.edge("domain_profile", "ragflow", "prompts, markers, top_k")
    g.edge("domain_profile", "loader", "chunk sizes, strategy")
    g.edge("domain_profile", "figures", "triage thresholds, vision")
    g.edge("routes", "agent")
    g.edge("routes", "loader")
    g.edge("agent", "ragflow")
    g.edge("ragflow", "milvus", "hybrid search")
    g.edge("ragflow", "postgres", "auto-merge parents")
    g.edge("ragflow", "redis", "cache reads")
    g.edge("loader", "figures", "image blocks")
    g.edge("loader", "milvus", "leaf chunks")
    g.edge("loader", "postgres", "parent chunks")
    g.edge("figures", "blobs", "raw image bytes")
    g.edge("figures", "postgres", "dossiers + extraction cache")
    g.edge("domain_profile", "delivery", "delivery policy, budgets")
    g.edge("agent", "delivery", "view_figure, surfaced assets")
    g.edge("delivery", "blobs", "reads image bytes")
    g.edge("delivery", "postgres", "dossier lookup")
    return g


def build_delivery_graph() -> Graph:
    g = Graph(
        "delivery", "Asset delivery — one backend, any client",
        "How a retrieved image reaches a consumer, and why replacing the UI needs no "
        "backend change.",
    )
    g.cluster("turn", "During a chat turn")
    g.cluster("escalate", "view_figure — the escalation ladder")
    g.cluster("render", "AssetPresenter — capability-driven rendition")
    g.cluster("clients", "Consumers (each declares its own ClientCapabilities)")

    g.node("retrieve", "search_knowledge_base\nchunk text includes the asset_id", "step", "turn")
    g.node("pin", "ctx.note_surfaced_assets\npins ids on SessionAssetState", "asset", "turn")
    g.node("enough", "does the caption / transcription\nanswer the question?", "decision", "turn")

    g.node("observed", "1. prior observations\nthis session", "asset", "escalate")
    g.node("dossier_text", "2. dossier text surface\ncaption + description + transcription", "asset", "escalate")
    g.node("pixels", "3. pixel read (budgeted)\nFigureReader -> vision model", "llm", "escalate")
    g.node("record", "record_observation\nimage becomes text for the rest\nof the conversation", "asset", "escalate")

    g.node("capabilities", "effective_capabilities\nclient wishes clamped by profile policy", "step", "render")
    g.node("present", "AssetPresenter.present_many\nresolve_mode(content_type, byte_size)", "step", "render")
    g.node("reference", "AssetReference (JSON)\nasset_id · url | inline_data\ncaption · alt_text · source", "asset", "render")

    g.node("browser", "Browser\naccepts_images -> url", "ok", "clients")
    g.node("bot", "Bot / webhook\nprefers_inline -> data URI", "ok", "clients")
    g.node("headless", "CLI / logs / audit\naccepts_images=false -> metadata", "ok", "clients")

    g.node("answer", "Chat response\nassets[] alongside rag_trace\n(SSE: its own 'assets' event)", "ok")
    g.node("bytes_route", "GET /media/{asset_id}\nauth · sha256 ETag · immutable cache", "store")
    g.node("resolve_route", "POST /media/resolve\nbatch, capability-aware\n(the integration entry point)", "store")

    g.edge("retrieve", "pin")
    g.edge("pin", "enough")
    g.edge("enough", "capabilities", "yes — answer from text")
    g.edge("enough", "observed", "no — call view_figure")
    g.edge("observed", "capabilities", "already known", dashed=True)
    g.edge("observed", "dossier_text", "nothing recorded yet")
    g.edge("dossier_text", "pixels", "still insufficient\nand budget remains")
    g.edge("pixels", "record")
    g.edge("record", "capabilities")
    g.edge("capabilities", "present")
    g.edge("present", "reference")
    g.edge("reference", "answer")
    g.edge("answer", "browser")
    g.edge("answer", "bot")
    g.edge("answer", "headless")
    g.edge("browser", "bytes_route", "fetches url")
    g.edge("bot", "resolve_route", "or resolves later", dashed=True)
    return g


BUILDERS = {
    "ingest": build_ingest_graph,
    "query": build_query_graph,
    "delivery": build_delivery_graph,
    "architecture": build_architecture_graph,
}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #fafafa; color: #111; }}
  header {{ padding: 1rem 1.5rem; background: #fff; border-bottom: 1px solid #ddd; }}
  h1 {{ font-size: 1.15rem; margin: 0 0 .25rem; }}
  p {{ margin: 0; color: #555; font-size: .9rem; }}
  nav {{ padding: .75rem 1.5rem; background:#fff; border-bottom:1px solid #eee; }}
  nav a {{ margin-right: 1rem; color:#2C5F9E; text-decoration:none; font-size:.9rem; }}
  section {{ padding: 1.5rem; }}
  .diagram {{ background:#fff; border:1px solid #e3e3e3; border-radius:6px; padding:1rem; overflow-x:auto; }}
  .fallback {{ color:#9E3A3A; font-size:.85rem; }}
</style>
<header>
  <h1>{title}</h1>
  <p>Generated by draw_rag_graph.py — re-run it after changing the pipelines.</p>
</header>
<nav>{nav}</nav>
{sections}
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  mermaid.initialize({{ startOnLoad: true, maxTextSize: 200000, flowchart: {{ useMaxWidth: false }} }});
</script>
<noscript><p class="fallback">This viewer needs JavaScript. The .md and .mmd files next to it
contain the same diagrams as text.</p></noscript>
"""


def write_outputs(graphs: List[Graph], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for graph in graphs:
        mermaid_source = render_mermaid(graph)

        mmd_path = out_dir / f"{graph.name}.mmd"
        mmd_path.write_text(mermaid_source + "\n", encoding="utf-8")
        written.append(mmd_path)

        md_path = out_dir / f"{graph.name}.md"
        md_path.write_text(
            f"# {graph.title}\n\n{graph.description}\n\n```mermaid\n{mermaid_source}\n```\n",
            encoding="utf-8",
        )
        written.append(md_path)

    sections = []
    nav = []
    for graph in graphs:
        nav.append(f'<a href="#{graph.name}">{graph.name}</a>')
        sections.append(
            f'<section id="{graph.name}">'
            f"<h2>{html.escape(graph.title)}</h2>"
            f"<p>{html.escape(graph.description)}</p>"
            f'<div class="diagram"><pre class="mermaid">{html.escape(render_mermaid(graph))}</pre></div>'
            f"</section>"
        )
    index = out_dir / "index.html"
    index.write_text(
        HTML_TEMPLATE.format(
            title="SuperMew pipelines" if len(graphs) > 1 else graphs[0].title,
            nav="".join(nav),
            sections="\n".join(sections),
        ),
        encoding="utf-8",
    )
    written.append(index)
    return written


def write_pngs(graphs: List[Graph], out_dir: Path) -> Tuple[List[Path], Optional[str]]:
    """PNGs via Graphviz. Returns (paths, skip_reason)."""
    if shutil.which("dot") is None:
        return [], (
            "Graphviz 'dot' is not on PATH, so PNGs were skipped. "
            "The Mermaid outputs above are complete on their own. "
            "To enable PNGs: winget install Graphviz.Graphviz (Windows), "
            "brew install graphviz (macOS), apt install graphviz (Debian/Ubuntu)."
        )
    try:
        import graphviz  # noqa: F401
    except ImportError:
        return [], "The `graphviz` Python package is not installed (uv sync --group dev)."

    written: List[Path] = []
    for graph in graphs:
        dot = render_graphviz(graph)
        if dot is None:
            continue
        try:
            rendered = dot.render(filename=str(out_dir / graph.name), cleanup=True)
            written.append(Path(rendered))
        except (subprocess.CalledProcessError, OSError) as exc:
            return written, f"Graphviz failed while rendering {graph.name}: {exc}"
    return written, None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", choices=sorted(BUILDERS), action="append",
                        help="Render only this view (repeatable). Default: all three.")
    parser.add_argument("--out", default="diagrams", type=Path, help="Output directory (default: ./diagrams)")
    parser.add_argument("--no-png", action="store_true", help="Skip Graphviz PNGs even when dot is available.")
    parser.add_argument("--print", dest="print_only", action="store_true",
                        help="Print the graph(s) as text to stdout and write no files.")
    args = parser.parse_args(argv)

    names = args.view or list(BUILDERS)
    graphs = [BUILDERS[name]() for name in names]

    for graph in graphs:
        graph.validate()

    if args.print_only:
        # A Windows console defaults to a legacy codepage, and these labels carry em
        # dashes and arrows. Without this the whole point of a printable view — that
        # you can read it where you already are — is lost to mojibake.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):  # pragma: no cover - very old or odd stdout
            pass
        for graph in graphs:
            print(render_text(graph))
        return 0

    written = write_outputs(graphs, args.out)
    print(f"Rendered {len(graphs)} view(s) to {args.out}/")
    for path in written:
        print(f"  {path}")

    if not args.no_png:
        pngs, skip_reason = write_pngs(graphs, args.out)
        for path in pngs:
            print(f"  {path}")
        if skip_reason:
            print(f"\nNote: {skip_reason}")

    print(
        f"\nView it:"
        f"\n  VS Code — open {args.out}/query.md and press Ctrl+Shift+V (Mermaid renders natively)"
        f"\n  Browser — open {args.out}/index.html"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
