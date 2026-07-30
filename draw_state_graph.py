"""Render the RAG state machine — nodes, connections, and state — to a single PNG.

The graph is read straight out of `backend/rag/pipeline.py` with `ast`:

  * `add_node(...)`                  -> nodes (id + the handler bound to it)
  * `set_entry_point` / `add_edge`   -> solid edges
  * `add_conditional_edges(...)`     -> dashed edges, labelled with the router's route keys
  * a router returning `Send(...)`   -> fan-out edge (parallel branches)
  * the `StateGraph(...)` state class -> the state panel on the right

Nothing is imported, so a run takes a second: no models load, no network, no env.
Drawing is pure Pillow, so no Graphviz `dot` binary and no browser is needed either.

Usage:
    uv run python draw_state_graph.py                      # -> diagrams/rag_state_graph.png
    uv run python draw_state_graph.py --out build/rag.png
    uv run python draw_state_graph.py --scale 3            # bigger/crisper raster
    uv run python draw_state_graph.py --check              # diff against the compiled graph
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

DEFAULT_SOURCE = Path("backend/rag/pipeline.py")
DEFAULT_OUT = Path("diagrams/rag_state_graph.png")

START, END = "__start__", "__end__"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class Node:
    id: str
    handler: str = ""          # the function registered for this node
    router: str = ""           # the function that picks this node's outgoing branch
    role: str = "step"         # key into ROLES, drives colour only


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    conditional: bool = False  # came from add_conditional_edges -> drawn dashed
    fanout: bool = False       # Send() dispatch -> drawn as a parallel branch


@dataclass
class StateField:
    name: str
    annotation: str
    reducer: str = ""          # Annotated[..., reducer]; parallel writes merge here


@dataclass
class Pipeline:
    state_name: str
    builder: str
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    state: List[StateField] = field(default_factory=list)

    def node_ids(self) -> List[str]:
        return [node.id for node in self.nodes]


# role -> (fill, stroke). Matched against the node id, first hit wins.
ROLES: Dict[str, Tuple[str, str]] = {
    "entry":     ("#E8E8E8", "#4A4A4A"),
    "terminal":  ("#E4E7EB", "#55606B"),
    "gate":      ("#FCF3CF", "#9A7D0A"),
    "llm":       ("#FDE9D9", "#C0722C"),
    "retrieval": ("#E6E0F8", "#5B4B9E"),
    "assess":    ("#F9E0EA", "#9E3A6B"),
    "parallel":  ("#DFF3EC", "#2E7D63"),
    "merge":     ("#E3F0DC", "#4E7A34"),
    "step":      ("#DCE9F8", "#2C5F9E"),
}

ROLE_RULES: Sequence[Tuple[str, str]] = (
    ("gate", "gate"),
    ("retriev", "retrieval"),
    ("recall", "retrieval"),
    ("grade", "assess"),
    ("classif", "llm"),
    ("rewrit", "llm"),
    ("sub_agent", "parallel"),
    ("synthes", "merge"),
)


def role_for(node_id: str) -> str:
    if node_id == START:
        return "entry"
    if node_id == END:
        return "terminal"
    for needle, role in ROLE_RULES:
        if needle in node_id:
            return role
    return "step"


# ---------------------------------------------------------------------------
# Extraction — read the wiring out of the source, never guess at it
# ---------------------------------------------------------------------------

WIRING_CALLS = {"add_node", "add_edge", "add_conditional_edges", "set_entry_point", "set_finish_point"}


def _literal(node: ast.AST) -> Optional[str]:
    """`"retrieve_initial"` -> that string; `END` / `START` -> the sentinel ids."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return {"END": END, "START": START}.get(node.id)
    if isinstance(node, ast.Attribute):  # e.g. graph.END
        return {"END": END, "START": START}.get(node.attr)
    return None


def _name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _calls_in_order(tree: ast.AST) -> List[ast.Call]:
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    return sorted(calls, key=lambda c: (c.lineno, c.col_offset))


def _routes_from_router(func: ast.FunctionDef) -> Tuple[List[str], List[str]]:
    """(route keys, Send targets) for a conditional-edge router with no mapping dict."""
    sends = [
        target
        for call in _calls_in_order(func)
        if _name_of(call.func) == "Send" and call.args
        for target in [_literal(call.args[0])]
        if target
    ]
    keys: List[str] = []
    annotation = func.returns
    if isinstance(annotation, ast.Subscript) and _name_of(annotation.value) == "Literal":
        literals = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
        keys = [value for value in (_literal(el) for el in literals) if value]
    if not keys:  # fall back to whatever strings the body returns
        keys = [
            value
            for stmt in ast.walk(func)
            if isinstance(stmt, ast.Return) and stmt.value is not None
            for value in [_literal(stmt.value)]
            if value
        ]
    return list(dict.fromkeys(keys)), list(dict.fromkeys(sends))


def _state_fields(tree: ast.Module, state_name: str) -> List[StateField]:
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        if cls.name != state_name:
            continue
        fields: List[StateField] = []
        for stmt in cls.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            annotation, reducer = stmt.annotation, ""
            if isinstance(annotation, ast.Subscript) and _name_of(annotation.value) == "Annotated":
                parts = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
                if parts:
                    reducer = ast.unparse(parts[1]) if len(parts) > 1 else ""
                    annotation = parts[0]
            fields.append(StateField(stmt.target.id, ast.unparse(annotation), reducer))
        return fields
    return []


def extract(source_path: Path) -> Pipeline:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    builder_fn: Optional[ast.FunctionDef] = None
    state_name = ""
    for func in functions.values():
        for call in _calls_in_order(func):
            if _name_of(call.func) == "StateGraph":
                builder_fn, state_name = func, (_name_of(call.args[0]) if call.args else "State")
                break
        if builder_fn:
            break
    if builder_fn is None:
        raise SystemExit(f"{source_path}: no StateGraph(...) construction found")

    pipeline = Pipeline(state_name=state_name, builder=builder_fn.name)
    seen: Dict[str, Node] = {}

    def ensure(node_id: str) -> Node:
        if node_id not in seen:
            node = Node(node_id, role=role_for(node_id))
            seen[node_id] = node
            pipeline.nodes.append(node)
        return seen[node_id]

    for call in _calls_in_order(builder_fn):
        method = _name_of(call.func)
        if method not in WIRING_CALLS or not call.args:
            continue
        first = _literal(call.args[0])

        if method == "add_node" and first:
            node = ensure(first)
            node.handler = _name_of(call.args[1]) if len(call.args) > 1 else ""
        elif method == "set_entry_point" and first:
            ensure(START)
            ensure(first)
            pipeline.edges.append(Edge(START, first))
        elif method == "set_finish_point" and first:
            ensure(END)
            pipeline.edges.append(Edge(first, END))
        elif method == "add_edge" and first and len(call.args) > 1:
            target = _literal(call.args[1])
            if target:
                ensure(first)
                ensure(target)
                pipeline.edges.append(Edge(first, target))
        elif method == "add_conditional_edges" and first:
            source = ensure(first)
            source.router = _name_of(call.args[1]) if len(call.args) > 1 else ""
            mapping = call.args[2] if len(call.args) > 2 else None
            if isinstance(mapping, ast.Dict):
                for key, value in zip(mapping.keys, mapping.values):
                    target, label = _literal(value), _literal(key) or ""
                    if target:
                        ensure(target)
                        # A route key that just repeats the target name says nothing the
                        # arrow does not already say, so it is left off the diagram.
                        pipeline.edges.append(
                            Edge(first, target, "" if label == target else label, conditional=True)
                        )
                continue
            router = functions.get(_name_of(call.args[1])) if len(call.args) > 1 else None
            if router is None:
                continue
            keys, sends = _routes_from_router(router)
            for target in sends:
                ensure(target)
                pipeline.edges.append(
                    Edge(first, target, "Send() fan-out\none branch per sub-question", conditional=True, fanout=True)
                )
            if not sends:
                for key in keys:
                    target = END if key == "end" else key
                    ensure(target)
                    pipeline.edges.append(
                        Edge(first, target, "" if key == target else key, conditional=True)
                    )

    pipeline.state = _state_fields(tree, state_name)
    return pipeline


# ---------------------------------------------------------------------------
# Layout — layered top-down, with side channels for the awkward edges
# ---------------------------------------------------------------------------

DIRECT, LONG, BACK = "direct", "long", "back"


def _back_edge_indices(pipeline: Pipeline) -> set[int]:
    """Edges that close a cycle (the rewrite loop), found by DFS colouring."""
    adjacency: Dict[str, List[Tuple[str, int]]] = {node.id: [] for node in pipeline.nodes}
    for index, edge in enumerate(pipeline.edges):
        adjacency[edge.src].append((edge.dst, index))

    white, grey, black = 0, 1, 2
    colour = {node.id: white for node in pipeline.nodes}
    back: set[int] = set()

    def visit(root: str) -> None:
        colour[root] = grey
        stack = [(root, iter(adjacency[root]))]
        while stack:
            node, children = stack[-1]
            advanced = False
            for target, index in children:
                if colour[target] == grey:
                    back.add(index)
                elif colour[target] == white:
                    colour[target] = grey
                    stack.append((target, iter(adjacency[target])))
                    advanced = True
                    break
            if not advanced:
                colour[node] = black
                stack.pop()

    for root in ([START] if START in colour else []) + pipeline.node_ids():
        if colour[root] == white:
            visit(root)
    return back


def _layer_nodes(pipeline: Pipeline, back: set[int]) -> Dict[str, int]:
    """Longest-path layering over the acyclic part; END is pinned to the bottom."""
    forward = [edge for index, edge in enumerate(pipeline.edges) if index not in back]
    layer = {node.id: 0 for node in pipeline.nodes}
    for _ in range(len(pipeline.nodes) + 1):
        changed = False
        for edge in forward:
            if layer[edge.dst] < layer[edge.src] + 1:
                layer[edge.dst] = layer[edge.src] + 1
                changed = True
        if not changed:
            break
    if END in layer:
        layer[END] = max([value for key, value in layer.items() if key != END] or [0]) + 1
    return layer


def _order_within_layers(pipeline: Pipeline, layer: Dict[str, int]) -> Dict[int, List[str]]:
    rows: Dict[int, List[str]] = {}
    for node in pipeline.nodes:  # declaration order is a sane starting point
        rows.setdefault(layer[node.id], []).append(node.id)

    predecessors: Dict[str, List[str]] = {node.id: [] for node in pipeline.nodes}
    for edge in pipeline.edges:
        predecessors[edge.dst].append(edge.src)

    for _ in range(3):  # barycentre sweeps: pull each node under its parents
        position = {node_id: index for row in rows.values() for index, node_id in enumerate(row)}
        for depth in sorted(rows):
            if depth == 0:
                continue
            row = rows[depth]
            rows[depth] = sorted(
                row,
                key=lambda node_id: (
                    sum(position[parent] for parent in predecessors[node_id] if layer[parent] < depth)
                    / max(1, len([p for p in predecessors[node_id] if layer[p] < depth])),
                    row.index(node_id),
                ),
            )
    return rows


@dataclass
class Box:
    node: Node
    lines: List[str]
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def bottom(self) -> float:
        return self.y + self.h


@dataclass
class Route:
    edge: Edge
    kind: str
    points: List[Tuple[float, float]]
    label_at: Tuple[float, float]
    label_anchor: str = "start"  # "start" draws right of the point, "end" left of it


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else (
        "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _bezier(p0, p1, p2, p3, steps: int = 40) -> List[Tuple[float, float]]:
    points = []
    for step in range(steps + 1):
        t = step / steps
        u = 1 - t
        points.append((
            u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1],
        ))
    return points


def _rounded_path(corners, radius: float = 14, steps: int = 8) -> List[Tuple[float, float]]:
    """Polyline with the corners filleted — used for gutter routes, where the straight
    run has to land exactly on the lane instead of merely leaning towards it."""
    def towards(origin, target, distance):
        dx, dy = target[0] - origin[0], target[1] - origin[1]
        length = math.hypot(dx, dy) or 1.0
        step = min(distance, length / 2)
        return (origin[0] + dx / length * step, origin[1] + dy / length * step)

    path = [corners[0]]
    for previous, corner, following in zip(corners, corners[1:], corners[2:]):
        entry, exit_ = towards(corner, previous, radius), towards(corner, following, radius)
        path.append(entry)
        for step in range(1, steps):
            t = step / steps
            u = 1 - t
            path.append((u * u * entry[0] + 2 * u * t * corner[0] + t * t * exit_[0],
                         u * u * entry[1] + 2 * u * t * corner[1] + t * t * exit_[1]))
        path.append(exit_)
    path.append(corners[-1])
    return path


def _dashed_line(draw, points, fill, width, dash=10, gap=7) -> None:
    carry, drawing = 0.0, True
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        segment = math.hypot(x2 - x1, y2 - y1)
        travelled = 0.0
        while travelled < segment:
            span = min((dash if drawing else gap) - carry, segment - travelled)
            if drawing:
                t0, t1 = travelled / segment, (travelled + span) / segment
                draw.line(
                    [(x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0),
                     (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)],
                    fill=fill, width=width,
                )
            carry += span
            travelled += span
            if carry >= (dash if drawing else gap) - 0.01:
                drawing, carry = not drawing, 0.0


def _arrow_head(draw, tip, tail, fill, size) -> None:
    dx, dy = tip[0] - tail[0], tip[1] - tail[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    base = (tip[0] - ux * size, tip[1] - uy * size)
    draw.polygon(
        [tip,
         (base[0] - uy * size * 0.42, base[1] + ux * size * 0.42),
         (base[0] + uy * size * 0.42, base[1] - ux * size * 0.42)],
        fill=fill,
    )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

INK = "#1B1B1B"
MUTED = "#666666"
LINE = "#5A6472"
FANOUT_INK = "#2E7D63"
CARD = "#FFFFFF"
PAPER = "#F7F8FA"
RULE = "#DEE2E8"


class Renderer:
    """Lays the graph out in unscaled points, then draws it at `scale`."""

    MARGIN = 34
    H_GAP = 40
    V_GAP = 74
    NODE_PAD_X = 16
    NODE_PAD_Y = 12
    LANE = 26
    HEADER = 96
    PANEL_GAP = 34

    def __init__(self, pipeline: Pipeline, scale: int = 2, source_label: str = "") -> None:
        self.pipeline = pipeline
        self.scale = scale
        self.source_label = source_label or "source"
        probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        self.f_title = _font(22 * scale, bold=True)
        self.f_sub = _font(12 * scale)
        self.f_node = _font(13 * scale, bold=True)
        self.f_small = _font(10 * scale)
        self.f_edge = _font(10 * scale)
        self.f_panel = _font(11 * scale)
        self.f_panel_head = _font(13 * scale, bold=True)
        self.probe = probe

    # -- measurement uses the scaled fonts, so keep one conversion in one place
    def _w(self, text: str, font) -> float:
        return _text_size(self.probe, text, font)[0] / self.scale

    def _h(self, font) -> float:
        return _text_size(self.probe, "Ag", font)[1] / self.scale

    # -- layout ------------------------------------------------------------
    def layout(self) -> None:
        pipeline = self.pipeline
        back = _back_edge_indices(pipeline)
        self.layer = _layer_nodes(pipeline, back)
        self.rows = _order_within_layers(pipeline, self.layer)

        line_h = self._h(self.f_node) + 4
        small_h = self._h(self.f_small) + 3

        self.boxes: Dict[str, Box] = {}
        for node in pipeline.nodes:
            label = {START: "START", END: "END"}.get(node.id, node.id)
            lines = [label]
            if node.handler and node.handler != node.id:
                lines.append(f"→ {node.handler}()")
            if node.router:
                lines.append(f"? {node.router}()")
            box = Box(node, lines)
            widths = [self._w(lines[0], self.f_node)]
            widths += [self._w(rest, self.f_small) for rest in lines[1:]]
            box.w = max(widths) + 2 * self.NODE_PAD_X
            box.h = line_h + small_h * (len(lines) - 1) + 2 * self.NODE_PAD_Y
            self.boxes[node.id] = box

        node_w = max(box.w for box in self.boxes.values())
        row_width = max(
            len(row) * node_w + (len(row) - 1) * self.H_GAP for row in self.rows.values()
        )

        # lanes: long forward edges run down the left gutter, back edges up the right
        self.routes_meta: List[Tuple[int, str]] = []
        for index, edge in enumerate(pipeline.edges):
            span = self.layer[edge.dst] - self.layer[edge.src]
            if index in back or span <= 0:
                self.routes_meta.append((index, BACK))
            elif span == 1:
                self.routes_meta.append((index, DIRECT))
            else:
                self.routes_meta.append((index, LONG))

        left_lanes = sum(1 for _, kind in self.routes_meta if kind == LONG)
        right_lanes = sum(1 for _, kind in self.routes_meta if kind == BACK)
        # Gutter-routed edges carry their label inside the gutter, so it has to fit there.
        gutter_label_w = max(
            [self._w(self.pipeline.edges[i].label.split("\n")[0], self.f_edge) + 16
             for i, kind in self.routes_meta if kind in (LONG, BACK) and self.pipeline.edges[i].label]
            or [0]
        )
        self.left_gutter = left_lanes * self.LANE + (14 + gutter_label_w if left_lanes else 0)
        self.right_gutter = right_lanes * self.LANE + (14 + gutter_label_w if right_lanes else 0)

        origin_x = self.MARGIN + self.left_gutter
        y = self.MARGIN + self.HEADER
        for depth in sorted(self.rows):
            row = self.rows[depth]
            row_w = len(row) * node_w + (len(row) - 1) * self.H_GAP
            x = origin_x + (row_width - row_w) / 2
            tallest = 0.0
            for node_id in row:
                box = self.boxes[node_id]
                box.w, box.x, box.y = node_w, x, y
                x += node_w + self.H_GAP
                tallest = max(tallest, box.h)
            y += tallest + self.V_GAP
        self.graph_bottom = y - self.V_GAP
        self.graph_right = origin_x + row_width

        self.routes = self._route_edges()
        self.panel_x = self.graph_right + self.right_gutter + self.PANEL_GAP
        self.panel_w, self.panel_h = self._panel_size()
        self.width = max(self.panel_x + self.panel_w, self.MARGIN + self._legend_width()) + self.MARGIN
        self.height = max(self.graph_bottom, self.MARGIN + self.HEADER + self.panel_h) + self.MARGIN + 54

    def _legend_width(self) -> float:
        roles = list(dict.fromkeys(node.role for node in self.pipeline.nodes))
        width = sum(45 + self._w(role, self.f_small) for role in roles)
        width += sum(50 + self._w(text, self.f_small) for text in self.EDGE_LEGEND)
        return width

    def _ports(self) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """Spread parallel edges across a node's edge instead of stacking them."""
        out_ports: Dict[str, List[int]] = {}
        in_ports: Dict[str, List[int]] = {}
        for index, (_, kind) in enumerate(self.routes_meta):
            edge = self.pipeline.edges[index]
            if kind == DIRECT:
                out_ports.setdefault(edge.src, []).append(index)
                in_ports.setdefault(edge.dst, []).append(index)
        for indices in out_ports.values():  # left-to-right by where the edge is going
            indices.sort(key=lambda i: self.boxes[self.pipeline.edges[i].dst].cx)
        for indices in in_ports.values():   # and by where it came from
            indices.sort(key=lambda i: self.boxes[self.pipeline.edges[i].src].cx)
        return out_ports, in_ports

    def _route_edges(self) -> List[Route]:
        out_ports, in_ports = self._ports()
        left_lane = right_lane = 0
        routes: List[Route] = []

        for index, kind in self.routes_meta:
            edge = self.pipeline.edges[index]
            src, dst = self.boxes[edge.src], self.boxes[edge.dst]

            if kind == DIRECT:
                siblings = out_ports[edge.src]
                slot = (siblings.index(index) + 1) / (len(siblings) + 1)
                start = (src.x + src.w * slot, src.bottom)
                siblings_in = in_ports[edge.dst]
                slot_in = (siblings_in.index(index) + 1) / (len(siblings_in) + 1)
                finish = (dst.x + dst.w * slot_in, dst.y)
                mid = (start[1] + finish[1]) / 2
                points = _bezier(start, (start[0], mid), (finish[0], mid), finish)
                label_at, anchor = (points[len(points) // 2][0] + 11, mid), "start"
            elif kind == LONG:  # skips a layer — down the left gutter, clear of the boxes
                left_lane += 1
                lane_x = self.MARGIN + self.left_gutter - 14 - left_lane * self.LANE
                # Leave through the bottom and turn inside the row gap: a sideways exit
                # would run straight through whatever box sits to the left.
                exit_x = src.x + src.w * 0.28
                gap_y = src.bottom + 20
                finish = (dst.x, dst.y + dst.h * 0.42)
                points = _rounded_path([
                    (exit_x, src.bottom), (exit_x, gap_y),
                    (lane_x, gap_y), (lane_x, finish[1]), finish,
                ], radius=12)
                label_at, anchor = (lane_x - 6, (gap_y + finish[1]) / 2), "end"
            else:  # BACK — the loop, up the right gutter
                right_lane += 1
                lane_x = self.graph_right + 14 + right_lane * self.LANE
                start = (src.x + src.w, src.y + src.h * 0.38)
                finish = (dst.x + dst.w, dst.y + dst.h * 0.62)
                points = _rounded_path([start, (lane_x, start[1]), (lane_x, finish[1]), finish])
                label_at, anchor = (lane_x + 6, (start[1] + finish[1]) / 2), "start"

            routes.append(Route(edge, kind, points, label_at, anchor))
        return routes

    PANEL_ROWS_PER_COLUMN = 16

    def _panel_rows(self) -> List[Tuple[str, str]]:
        return [
            (field.name, field.annotation + (f"   += {field.reducer}" if field.reducer else ""))
            for field in self.pipeline.state
        ]

    def _panel_columns(self) -> List[List[Tuple[str, str]]]:
        """One column normally; split in two rather than grow a very tall page."""
        rows = self._panel_rows()
        if len(rows) <= self.PANEL_ROWS_PER_COLUMN:
            return [rows]
        half = (len(rows) + 1) // 2
        return [rows[:half], rows[half:]]

    def _panel_size(self) -> Tuple[float, float]:
        columns = self._panel_columns()
        self.panel_columns = columns
        self.panel_col_metrics = []
        for column in columns:
            name_w = max([self._w(name, self.f_panel_head) for name, _ in column] or [40])
            type_w = max([self._w(annotation, self.f_panel) for _, annotation in column] or [40])
            self.panel_col_metrics.append((name_w, name_w + 16 + type_w))
        head = self._h(self.f_panel_head) + 6
        row_h = self._h(self.f_panel) + 9
        tallest = max(len(column) for column in columns)
        width = 36 + sum(col_w for _, col_w in self.panel_col_metrics) + 30 * (len(columns) - 1)
        height = 24 + head + 14 + self._h(self.f_small) + tallest * row_h + 50
        return max(width, 260), height

    # -- drawing -----------------------------------------------------------
    def render(self) -> Image.Image:
        self.layout()
        s = self.scale
        image = Image.new("RGB", (int(self.width * s), int(self.height * s)), PAPER)
        draw = ImageDraw.Draw(image)

        def S(point: Tuple[float, float]) -> Tuple[float, float]:
            return (point[0] * s, point[1] * s)

        self._draw_header(draw, s)
        for route in self.routes:
            self._draw_edge(draw, route, S, s)
        for box in self.boxes.values():
            self._draw_box(draw, box, s)
        self._draw_panel(draw, s)
        self._draw_legend(draw, s)
        return image

    def _draw_header(self, draw, s) -> None:
        pipeline = self.pipeline
        x, y = self.MARGIN * s, self.MARGIN * s
        draw.text((x, y), f"{pipeline.state_name} graph — {pipeline.builder}()", font=self.f_title, fill=INK)
        y += _text_size(draw, "Ag", self.f_title)[1] + 14 * s
        subtitle = (
            f"{len(pipeline.nodes)} nodes · {len(pipeline.edges)} edges · "
            f"{len(pipeline.state)} state channels — extracted from {self.source_label}"
        )
        draw.text((x, y), subtitle, font=self.f_sub, fill=MUTED)
        y += _text_size(draw, "Ag", self.f_sub)[1] + 10 * s
        draw.line([(x, y), ((self.width - self.MARGIN) * s, y)], fill=RULE, width=max(1, s))

    def _draw_edge(self, draw, route: Route, S, s) -> None:
        colour = FANOUT_INK if route.edge.fanout else LINE
        width = max(1, int(1.6 * s))
        points = [S(point) for point in route.points]
        if route.edge.conditional:
            _dashed_line(draw, points, colour, width, dash=9 * s, gap=6 * s)
        else:
            draw.line(points, fill=colour, width=width, joint="curve")
        _arrow_head(draw, points[-1], points[-4], colour, 9 * s)

        if not route.edge.label:
            return
        lines = route.edge.label.split("\n")
        line_h = _text_size(draw, "Ag", self.f_edge)[1] + 3 * s
        text_w = max(_text_size(draw, line, self.f_edge)[0] for line in lines)
        left, top = S(route.label_at)
        if route.label_anchor == "end":
            left -= text_w
        pad = 5 * s
        draw.rounded_rectangle(
            [left - pad, top - pad - line_h * len(lines) / 2,
             left + text_w + pad, top + pad + line_h * len(lines) / 2],
            radius=4 * s, fill=CARD, outline=RULE, width=max(1, s // 2 or 1),
        )
        text_y = top - line_h * len(lines) / 2 + 2 * s
        for line in lines:
            draw.text((left, text_y), line, font=self.f_edge, fill=MUTED if not route.edge.fanout else FANOUT_INK)
            text_y += line_h

    def _draw_box(self, draw, box: Box, s) -> None:
        fill, stroke = ROLES[box.node.role]
        rounded = box.node.id in (START, END)
        rect = [box.x * s, box.y * s, (box.x + box.w) * s, box.bottom * s]
        draw.rounded_rectangle(
            rect, radius=(box.h / 2 if rounded else 8) * s,
            fill=fill, outline=stroke, width=max(1, int(1.6 * s)),
        )
        title_w, title_h = _text_size(draw, box.lines[0], self.f_node)
        centre_x = box.cx * s
        y = box.y * s + self.NODE_PAD_Y * s
        draw.text((centre_x - title_w / 2, y), box.lines[0], font=self.f_node, fill=INK)
        y += title_h + 5 * s
        for line in box.lines[1:]:
            sub_w, sub_h = _text_size(draw, line, self.f_small)
            draw.text((centre_x - sub_w / 2, y), line, font=self.f_small, fill=stroke)
            y += sub_h + 3 * s

    def _draw_panel(self, draw, s) -> None:
        x, y = self.panel_x, self.MARGIN + self.HEADER - 22
        draw.rounded_rectangle(
            [x * s, y * s, (x + self.panel_w) * s, (y + self.panel_h) * s],
            radius=10 * s, fill=CARD, outline=RULE, width=max(1, s),
        )
        text_x = (x + 18) * s
        cursor = (y + 16) * s
        draw.text((text_x, cursor), f"{self.pipeline.state_name}", font=self.f_panel_head, fill=INK)
        cursor += _text_size(draw, "Ag", self.f_panel_head)[1] + 6 * s
        draw.text((text_x, cursor), "the state every node reads and writes", font=self.f_small, fill=MUTED)
        cursor += _text_size(draw, "Ag", self.f_small)[1] + 12 * s
        draw.line([(text_x, cursor), ((x + self.panel_w - 18) * s, cursor)], fill=RULE, width=max(1, s // 2 or 1))
        cursor += 10 * s

        row_h = _text_size(draw, "Ag", self.f_panel)[1] + 9 * s
        column_x = text_x
        deepest = cursor
        for column, (name_w, col_w) in zip(self.panel_columns, self.panel_col_metrics):
            row_y = cursor
            for name, annotation in column:
                draw.text((column_x, row_y), name, font=self.f_panel_head, fill=INK)
                draw.text((column_x + (name_w + 16) * s, row_y), annotation, font=self.f_panel, fill=MUTED)
                row_y += row_h
            deepest = max(deepest, row_y)
            column_x += (col_w + 30) * s
        draw.text((text_x, deepest + 10 * s),
                  "+= reducer — parallel branches merge into this channel",
                  font=self.f_small, fill=FANOUT_INK)

    EDGE_LEGEND = ("static edge", "conditional edge", "Send() fan-out")

    def _draw_legend(self, draw, s) -> None:
        y = (self.height - self.MARGIN - 22) * s
        x = self.MARGIN * s
        swatch = 11 * s
        used = list(dict.fromkeys(node.role for node in self.pipeline.nodes))
        for role in used:
            fill, stroke = ROLES[role]
            draw.rounded_rectangle([x, y, x + swatch * 1.8, y + swatch], radius=3 * s,
                                   fill=fill, outline=stroke, width=max(1, s))
            draw.text((x + swatch * 2.3, y - 1 * s), role, font=self.f_small, fill=MUTED)
            x += swatch * 2.3 + _text_size(draw, role, self.f_small)[0] + 18 * s

        for text in self.EDGE_LEGEND:
            dashed = text != "static edge"
            colour = FANOUT_INK if "Send()" in text else LINE
            line = [(x, y + swatch / 2), (x + 26 * s, y + swatch / 2)]
            if dashed:
                _dashed_line(draw, line, colour, max(1, int(1.6 * s)), dash=7 * s, gap=5 * s)
            else:
                draw.line(line, fill=colour, width=max(1, int(1.6 * s)))
            _arrow_head(draw, line[1], line[0], colour, 7 * s)
            draw.text((x + 32 * s, y - 1 * s), text, font=self.f_small, fill=MUTED)
            x += 32 * s + _text_size(draw, text, self.f_small)[0] + 18 * s


# ---------------------------------------------------------------------------
# Optional drift check against the compiled graph
# ---------------------------------------------------------------------------


def check_against_runtime(pipeline: Pipeline) -> int:
    """Compare the parsed graph with the one LangGraph actually compiles.

    Slow: it imports the pipeline, which loads the embedding model. LangGraph's own
    drawable graph cannot see `Send()` targets — it shows a placeholder edge to END
    instead, and prunes everything only reachable through the fan-out — so those
    differences are reported as notes. Anything else is real drift.
    """
    try:
        from backend.rag.pipeline import build_rag_graph
    except Exception as exc:  # noqa: BLE001 — any import failure tells the same story
        print(f"check skipped: could not import backend.rag.pipeline ({exc})")
        return 0

    runtime = build_rag_graph().get_graph()
    runtime_nodes, runtime_edges = set(runtime.nodes), {(e.source, e.target) for e in runtime.edges}
    parsed_nodes, parsed_edges = set(pipeline.node_ids()), {(e.src, e.dst) for e in pipeline.edges}

    fanout_sources = {edge.src for edge in pipeline.edges if edge.fanout}
    outgoing: Dict[str, List[str]] = {}
    for edge in pipeline.edges:
        outgoing.setdefault(edge.src, []).append(edge.dst)
    pruned, queue = set(), [edge.dst for edge in pipeline.edges if edge.fanout]
    while queue:  # the Send() branch and everything downstream of it
        node_id = queue.pop()
        if node_id in pruned or node_id in (START, END):
            continue
        pruned.add(node_id)
        queue.extend(outgoing.get(node_id, []))

    problems = 0
    for missing in sorted(runtime_nodes - parsed_nodes):
        print(f"  drift: compiled graph has node {missing!r}, the diagram does not")
        problems += 1
    for extra in sorted(parsed_nodes - runtime_nodes):
        why = "reachable only via Send()" if extra in pruned else "diagram only"
        print(f"  note: node {extra!r} not in the compiled view ({why})")

    for missing in sorted(runtime_edges - parsed_edges):
        if missing[0] in fanout_sources and missing[1] == END:
            print(f"  note: {missing[0]} -> END is LangGraph's placeholder for the Send() fan-out")
            continue
        print(f"  drift: compiled edge {missing[0]} -> {missing[1]} is missing from the diagram")
        problems += 1
    for extra in sorted(parsed_edges - runtime_edges):
        if any(edge.fanout for edge in pipeline.edges if (edge.src, edge.dst) == extra):
            why = "Send() edge, invisible to the compiled view"
        elif extra[0] in pruned:
            why = "downstream of Send(), pruned from the compiled view"
        else:
            why = "diagram only"
        print(f"  note: {extra[0]} -> {extra[1]} ({why})")

    print("check: no drift" if not problems else f"check: {problems} difference(s)")
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help=f"module holding the StateGraph build (default: {DEFAULT_SOURCE})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"PNG to write (default: {DEFAULT_OUT})")
    parser.add_argument("--scale", type=int, default=2, choices=(1, 2, 3, 4),
                        help="raster supersampling; 3-4 for print (default: 2)")
    parser.add_argument("--check", action="store_true",
                        help="also import the pipeline and diff the compiled graph (slow)")
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"error: {args.source} not found", file=sys.stderr)
        return 1

    pipeline = extract(args.source)
    if not pipeline.nodes:
        print(f"error: no nodes found in {args.source}", file=sys.stderr)
        return 1

    renderer = Renderer(pipeline, scale=args.scale,
                        source_label=str(args.source).replace("\\", "/"))
    image = renderer.render()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out, "PNG", optimize=True)

    print(f"{args.out}  ({image.width}x{image.height})")
    print(f"  {len(pipeline.nodes)} nodes: {', '.join(pipeline.node_ids())}")
    print(f"  {len(pipeline.edges)} edges, {len(pipeline.state)} {pipeline.state_name} channels")

    if args.check:
        return 1 if check_against_runtime(pipeline) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
