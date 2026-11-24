#!/usr/bin/env python3
"""Rebuild the DataOps graph shown in ``skrub-dataops.qmd``.

The "Declare the pipeline" code block on the *Data Ops: a basic example* slide
is valid Python that builds the plan without touching any data, so it is the
single source of truth: this script extracts and executes it, draws the plan,
and splits the drawing into the cumulative stages used by the line-by-line
reveal.

It writes ``images/model_graph.svg`` (the full graph, also used on the
"Training and persisting the Data Ops" slide) and ``images/model_graph_1.svg``
... ``images/model_graph_5.svg``.

The Quarto ``pre-render`` hook in ``_quarto.yml`` runs this before every
render, so the graphs can never drift from the code on the slide.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import skrub
from skrub import selectors as s
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reveal_svg import (  # noqa: E402
    classes,
    local_name,
    parse_edge_endpoints,
    unwrap_anchors,
    visible_stage,
)

ROOT = Path(__file__).resolve().parents[1]
QMD = ROOT / "skrub-dataops.qmd"
IMAGES = ROOT / "images"

# The fenced block that follows "Declare the pipeline" on the slide.
BLOCK = re.compile(
    r"Declare the pipeline\s*\n+```\{\.python([^}]*)\}\n(.*?)\n```", re.DOTALL
)
LINE_NUMBERS = re.compile(r'code-line-numbers="([^"]*)"')

# Cumulative stages, aligned with the slide's code-line-numbers steps:
# "1-2|4|6-7|9|11-15". Each stage lists the nodes revealed at that step;
# edges are added automatically once both endpoints are visible.
STAGES = [
    ["node_0", "node_1"],  # 1-2    data_path -> read_csv
    ["node_3", "node_4"],  # 4      hist_path -> read_csv
    ["node_2", "node_7"],  # 6-7    X.drop / y["target"]
    ["node_5"],  # 9      X.merge(hist, ...)
    ["node_6", "node_8"],  # 11-15  StandardScaler -> LogisticRegression
]


def pipeline_block() -> tuple[str, str]:
    match = BLOCK.search(QMD.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(
            f"{QMD}: could not find the 'Declare the pipeline' code block"
        )
    return match.group(1), match.group(2)


def parse_step(step: str) -> list[int]:
    lines: list[int] = []
    for part in step.split(","):
        first, _, last = part.partition("-")
        lines.extend(range(int(first), int(last or first) + 1))
    return lines


def highlight_steps(attrs: str) -> list[list[int]]:
    match = LINE_NUMBERS.search(attrs)
    if match is None:
        raise SystemExit(f"{QMD}: the pipeline block has no code-line-numbers")
    raw = match.group(1)
    steps = raw.split("|")
    if any(not step.strip() for step in steps):
        raise SystemExit(
            f"{QMD}: code-line-numbers={raw!r} has an empty step; "
            "drop the leading/trailing '|'"
        )
    return [parse_step(step) for step in steps]


def check_highlight(code: str, steps: list[list[int]]) -> None:
    """Catch line ranges that drifted from the code block they refer to."""
    if len(steps) != len(STAGES):
        raise SystemExit(
            f"{QMD}: {len(steps)} highlight steps but {len(STAGES)} graph "
            "stages; update STAGES in this script"
        )
    code_lines = code.splitlines()
    for index, lines in enumerate(steps, start=1):
        for line in lines:
            if not 1 <= line <= len(code_lines):
                raise SystemExit(
                    f"{QMD}: step {index} highlights line {line}, but the "
                    f"code block has {len(code_lines)} lines"
                )
            if not code_lines[line - 1].strip():
                raise SystemExit(
                    f"{QMD}: step {index} highlights the blank line {line}; "
                    "adjust code-line-numbers"
                )


def draw_graph(code: str) -> bytes:
    namespace = {
        "pd": pd,
        "skrub": skrub,
        "s": s,
        "StandardScaler": StandardScaler,
        "LogisticRegression": LogisticRegression,
    }
    exec(code, namespace)
    try:
        prediction = namespace["pred"]
    except KeyError:
        raise SystemExit(
            f"{QMD}: the pipeline block should define the final plan as `pred`"
        ) from None
    return prediction.skb.draw_graph().svg


def check_stages(nodes: dict[str, ET.Element]) -> None:
    covered = {node for stage in STAGES for node in stage}
    missing = sorted(covered - nodes.keys())
    uncovered = sorted(nodes.keys() - covered)
    if missing or uncovered:
        raise SystemExit(
            "the graph no longer matches the reveal stages; update STAGES in "
            f"{Path(__file__).name}\n"
            f"  nodes not in the graph: {missing}\n"
            f"  nodes not revealed:    {uncovered}"
        )


def write_svg(path: Path, root: ET.Element) -> None:
    path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    attrs, code = pipeline_block()
    check_highlight(code, highlight_steps(attrs))

    raw_svg = draw_graph(code).decode("utf-8")
    root = ET.fromstring(raw_svg)
    graph = next(el for el in root if local_name(el.tag) == "g")

    # The <a xlink:title> wrappers embed a creation stack line with an absolute
    # path, which would make the committed SVGs differ between machines.
    unwrap_anchors(graph)
    ET.indent(root, space="  ")
    IMAGES.mkdir(exist_ok=True)
    write_svg(IMAGES / "model_graph.svg", root)

    nodes = {el.get("id"): el for el in graph if "node" in classes(el)}
    check_stages(nodes)

    endpoints = parse_edge_endpoints(raw_svg)
    revealed: set[str] = set()
    for index, stage in enumerate(STAGES, start=1):
        revealed.update(stage)
        stage_root = visible_stage(
            root,
            graph,
            revealed,
            endpoints,
            keep_background=False,
            keep_links=False,
            missing_edges=set(),
        )
        write_svg(IMAGES / f"model_graph_{index}.svg", stage_root)


if __name__ == "__main__":
    main()
