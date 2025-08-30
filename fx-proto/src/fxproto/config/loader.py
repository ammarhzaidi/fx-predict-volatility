# fx-proto/src/fxproto/config/loader.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional

import yaml

from fxproto.utils.paths import config_dir, data_root, outputs_root


# ---------- Dataclasses (typed config) ----------

@dataclass(frozen=True)
class DatesCfg:
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class PathsCfg:
    data_dir: str
    outputs_dir: str


@dataclass(frozen=True)
class SettingsCfg:
    pair: str
    horizon: int
    dates: DatesCfg
    paths: PathsCfg
    random_seed: int = 42


@dataclass(frozen=True)
class DataCfg:
    source: str
    interval: str
    symbol_map: Dict[str, str]
    columns: Optional[List[str]] = None


@dataclass(frozen=True)
class GraphNode:
    id: str
    type: Optional[str] = None


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    weight: float = 1.0


@dataclass(frozen=True)
class GraphScenarioShock:
    node: str
    delta: float


@dataclass(frozen=True)
class GraphScenario:
    id: str
    description: Optional[str]
    shocks: List[GraphScenarioShock]
    observe_nodes: Optional[List[str]] = None


@dataclass(frozen=True)
class GraphCfg:
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    scenarios: List[GraphScenario]


@dataclass(frozen=True)
class AppConfig:
    settings: SettingsCfg
    data: DataCfg
    graph: GraphCfg


# ---------- Helpers ----------

def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _to_settings(d: dict) -> SettingsCfg:
    dates = DatesCfg(**d["dates"])
    paths = PathsCfg(**d["paths"])
    return SettingsCfg(
        pair=str(d["pair"]),
        horizon=int(d["horizon"]),
        dates=dates,
        paths=paths,
        random_seed=int(d.get("random_seed", 42)),
    )


def _to_data(d: dict) -> DataCfg:
    return DataCfg(
        source=str(d["source"]),
        interval=str(d["interval"]),
        symbol_map=dict(d.get("symbol_map", {})),
        columns=d.get("columns"),
    )


def _to_graph(d: dict) -> GraphCfg:
    nodes = [GraphNode(**n) for n in d.get("nodes", [])]
    edges = [GraphEdge(**e) for e in d.get("edges", [])]
    scenarios: List[GraphScenario] = []
    for s in d.get("scenarios", []):
        shocks = [GraphScenarioShock(**x) for x in s.get("shocks", [])]
        scenarios.append(
            GraphScenario(
                id=s["id"],
                description=s.get("description"),
                shocks=shocks,
                observe_nodes=s.get("observe_nodes"),
            )
        )
    return GraphCfg(nodes=nodes, edges=edges, scenarios=scenarios)


# ---------- Public API ----------

@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache all YAML configs into a single typed config object."""
    # FIXED: Use the Path object directly, don't call it as function
    from fxproto.utils.paths import config_dir

    cfg_dir = config_dir  # It's already a Path object, don't call it!
    settings_raw = _read_yaml(cfg_dir / "settings.yaml")
    data_raw = _read_yaml(cfg_dir / "data.yaml")
    graph_raw = _read_yaml(cfg_dir / "graph.yaml")

    settings = _to_settings(settings_raw)
    data = _to_data(data_raw)
    graph = _to_graph(graph_raw)

    return AppConfig(settings=settings, data=data, graph=graph)


def _resolve_under_project(relative_or_abs: Path, fallback: Path) -> Path:
    """
    If `relative_or_abs` is absolute, return it.
    If relative, resolve it relative to the project's canonical roots.
    """
    p = Path(relative_or_abs)
    if p.is_absolute():
        return p
    # Resolve relative to the fx-proto root that utils.paths points to:
    # - data_root() -> <proj>/fx-proto/data
    # - outputs_root() -> <proj>/fx-proto/src/fxproto/outputs
    # So for a relative path like "fx-proto/data", join with fallback.parent
    return (fallback.parent / p).resolve()


def resolved_data_dir() -> Path:
    """
    Final absolute data directory based on settings.paths.data_dir,
    falling back to utils.paths.data_root() if not set.
    """
    cfg = get_config()
    desired = Path(cfg.settings.paths.data_dir)
    return _resolve_under_project(desired, data_root())


def resolved_outputs_dir() -> Path:
    """
    Final absolute outputs directory based on settings.paths.outputs_dir,
    falling back to utils.paths.outputs_root() if not set.
    """
    cfg = get_config()
    desired = Path(cfg.settings.paths.outputs_dir)
    return _resolve_under_project(desired, outputs_root())
