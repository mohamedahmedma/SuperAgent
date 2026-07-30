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
        "query", "Query pipeline — corrective RAG over the LangGraph state machine",
        "The LangGraph nodes plus the sub-steps that happen inside them.",
    )
    g.cluster("gate", "0. Domain gate — domain_gate_node (entry)")
    g.cluster("plan", "1. Complexity planning — classify_complexity")
    g.cluster("subagents", "3. Parallel sub-agents — rag_sub_agent x N via Send()")
    g.cluster("grade", "5. Evidence assessment — grade_documents_node")
    g.cluster("ladder", "The assessor ladder — cheapest rung first, stops at profile.rag.evidence_required_certainty")
    g.cluster("rewrite", "6. Query rewrite — rewrite_question -> retrieve_rewritten, max profile.rag.max_rewrites")
    g.cluster("resume", "HITL resume — resume_rag_from_hitl (separate entry point)")

    g.node("start", "User question\nsearch_knowledge_base tool\n-> run_rag_graph", "entry")
    g.node("gate_eligible", "gate eligible?\nskipped for follow-ups and short questions", "decision", "gate")
    g.node("gate_classify", "Shared query embedding\ncosine vs section reference vectors\nno model call, no search", "decision", "gate")
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
    g.node("rung1", "Rung 1 — lexical overlap\ncertainty LOW: reports, never concludes", "decision", "ladder")
    g.node("rung2", "Rung 2 — cross-encoder\ncalibrated per-chunk relevance\ncertainty MEDIUM", "decision", "ladder")
    g.node("rung3", "Rung 3 — GRADE_MODEL structured call\n-> EvidenceGrade + supporting_chunks\ncertainty HIGH", "llm", "ladder")
    g.node("report", "EvidenceReport\nper-chunk scores, certainty, provenance\nthe single source of truth", "step", "grade")
    g.node("route", "decide_route — pure policy\nrefuses to act below the required\ncertainty; max_rewrites governs rewrites", "decision", "grade")
    g.node("ctxpolicy", "select_context_indices — pure policy\ntrims only to chunks an assessment named\nrequires MEDIUM certainty", "decision", "grade")

    g.node("budget", "rewrite budget left?", "decision", "rewrite")
    g.node("rewriter", "FAST_MODEL structured call\n-> RewritePlan (exactly one method)", "llm", "rewrite")
    g.node("stepback", "Step-back\nabstract the question", "step", "rewrite")
    g.node("hyde", "HyDE\nhypothetical answer document\nretrieval aid only, never evidence", "step", "rewrite")
    filt_rewritten = _retrieval_cluster(g, "rewritten", "Retrieval pipeline (rewritten query)")

    g.node("answer", "Answer generation\nchunks formatted with [n] citations\nfigure chunks carry asset_ids", "ok")
    g.node("hitl_pause", "HITL pause\npending_hitl saved to the session\nprompt from profile.user_copy", "hitl")
    g.node("nokb", "No usable evidence\nprofile.user_copy.no_knowledge", "none")
    g.node("outage", "Retrieval error\nstatic retry notice — never no_knowledge", "none")

    g.node("hitl_answer", "User's follow-up answer\n+ original question", "entry", "resume")
    g.node("hitl_targeted", "Targeted retrieval\nskips planning and decomposition", "step", "resume")
    filt_hitl = _retrieval_cluster(g, "hitl", "Retrieval pipeline (HITL targeted)")

    g.edge("start", "gate_eligible")
    g.edge("gate_eligible", "gate_classify", "first-turn question")
    g.edge("gate_eligible", "fastpath", "skipped — abstains open", dashed=True)
    g.edge("gate_classify", "fastpath", "in domain (+ topic hints)")
    g.edge("gate_classify", "nokb", "out of domain — no search, no LLM", dashed=True)
    g.edge("fastpath", "complexity", "confidently simple")
    g.edge("fastpath", "planner", "not confident")
    g.edge("planner", "complexity")
    g.edge("complexity", "retrieve_initial", "simple")
    g.edge("complexity", "prepare_subs", "complex")
    g.edge("retrieve_initial", "recall_initial")
    g.edge(filt_initial, "rung0")
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
    g.edge("route", "hitl_pause", "clarify / scope_select", dashed=True)
    g.edge("route", "nokb", "no_knowledge", dashed=True)
    g.edge("route", "outage", "retrieval_error", dashed=True)
    g.edge("route", "budget", "rewrite")
    g.edge("budget", "nokb", "exhausted", dashed=True)
    g.edge("budget", "rewriter", "available")
    g.edge("rewriter", "stepback")
    g.edge("rewriter", "hyde")
    g.edge("stepback", "recall_rewritten")
    g.edge("hyde", "recall_rewritten")
    g.edge(filt_rewritten, "rung0", "re-assess", dashed=True)
    g.edge("hitl_answer", "hitl_targeted")
    g.edge("hitl_targeted", "recall_hitl")
    g.edge(filt_hitl, "rung0", "re-assess", dashed=True)
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
    args = parser.parse_args(argv)

    names = args.view or list(BUILDERS)
    graphs = [BUILDERS[name]() for name in names]

    for graph in graphs:
        graph.validate()

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
