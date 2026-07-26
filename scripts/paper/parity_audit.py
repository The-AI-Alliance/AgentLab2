#!/usr/bin/env python3
"""parity_audit.py — measure how faithfully each CUBE reproduces its upstream benchmark.

The audit answers a question reviewers asked directly: *is the CUBE wrapper the
same benchmark as the original implementation?* It does so statically, without
running an agent, by classifying each CUBE into one of three provenance classes
and then verifying the class-appropriate evidence:

  package    evaluation is delegated to the upstream authors' published package,
             so the scoring code is upstream's by construction. Verified by
             reading the declared dependency out of ``pyproject.toml``.
  vendored   evaluator source was copied at a pinned upstream commit. Verified
             by an *AST-level* comparison of every evaluator function against
             its upstream counterpart, which ignores formatting-only edits
             (quote style, trailing commas, line wrapping) and reports genuine
             semantic divergence.
  image      the benchmark's own container images and test commands are executed
             verbatim. Static evidence is the image reference; empirical
             evidence comes from the oracle run (``oracle_parity.py``).

Task-set identity (do we ship exactly the upstream task list, with byte-identical
instruction strings?) is checked for every CUBE whose upstream index is available.

Usage
-----
    # clone the upstream repos (~2 min) then audit
    scripts/paper/parity_audit.py --clone

    # reuse an existing clone directory
    scripts/paper/parity_audit.py --upstream-dir ~/parity-upstream

Outputs ``parity_audit.json`` plus a LaTeX table on stdout.
"""

from __future__ import annotations

import ast
import copy
import difflib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
CUBES = REPO_ROOT / "cubes"


# ─────────────────────────────────────────────────────────────────────────────
# Declarative audit spec: one entry per CUBE in the paper's corpus.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class UpstreamRepo:
    """An upstream benchmark repository to clone for comparison."""

    url: str
    commit: str | None = None  # None → default branch tip


@dataclass(frozen=True)
class CubeSpec:
    """How to audit one CUBE against its upstream implementation."""

    cube: str  # display name in the paper
    package_dir: str  # path under cubes/
    provenance: str  # "package" | "vendored" | "image"
    repo: UpstreamRepo | None = None
    # Some CUBEs vendor from more than one upstream: Windows Agent Arena is a
    # fork of OSWorld and shares its metric implementations, so its Windows
    # getters come from WAA while its metrics track OSWorld. Auditing every
    # directory against a single repo would report that split as divergence.
    extra_repos: tuple[tuple[str, UpstreamRepo], ...] = ()
    # vendored: (cube-relative dir, upstream-relative dir, upstream repo key)
    vendored_pairs: tuple[tuple[str, str, str], ...] = ()
    # task-set identity
    task_metadata: str | None = None  # cube-relative task_metadata.json
    upstream_tasks_glob: str | None = None  # upstream glob of per-task json
    task_id_key: str = "id"
    task_text_key: str | None = "instruction"
    cube_text_key: str = "abstract_description"
    # package provenance: dependency that carries upstream evaluation code
    upstream_package: str | None = None
    # image provenance: where the upstream container reference lives
    image_evidence: str | None = None


SPECS: tuple[CubeSpec, ...] = (
    CubeSpec(
        cube="OSWorld",
        package_dir="osworld-cube",
        provenance="vendored",
        repo=UpstreamRepo("https://github.com/xlang-ai/OSWorld.git", "e695a10"),
        vendored_pairs=(
            ("src/osworld_cube/vm_backend/getters", "desktop_env/evaluators/getters", "primary"),
            ("src/osworld_cube/vm_backend/metrics", "desktop_env/evaluators/metrics", "primary"),
        ),
        task_metadata="src/osworld_cube/task_metadata.json",
        upstream_tasks_glob="evaluation_examples/examples/*/*.json",
        cube_text_key="instruction",
    ),
    CubeSpec(
        cube="Windows Agent Arena",
        package_dir="windows-agent-arena-cube",
        provenance="vendored",
        repo=UpstreamRepo("https://github.com/microsoft/WindowsAgentArena.git"),
        extra_repos=(("osworld", UpstreamRepo("https://github.com/xlang-ai/OSWorld.git", "e695a10")),),
        vendored_pairs=(
            (
                "src/waa_cube/vm_backend/getters",
                "src/win-arena-container/client/desktop_env/evaluators/getters",
                "primary",
            ),
            ("src/waa_cube/vm_backend/metrics", "desktop_env/evaluators/metrics", "osworld"),
        ),
        task_metadata="src/waa_cube/task_metadata.json",
        upstream_tasks_glob="src/win-arena-container/client/evaluation_examples_windows/examples*/*/*.json",
    ),
    CubeSpec(
        cube="WebArena-Verified",
        package_dir="webarena-verified",
        provenance="package",
        upstream_package="webarena-verified",
    ),
    CubeSpec(
        cube="WorkArena",
        package_dir="workarena",
        provenance="package",
        upstream_package="browsergym-workarena",
    ),
    CubeSpec(
        cube="MiniWoB",
        package_dir="miniwob",
        provenance="package",
        upstream_package="miniwob",
    ),
    CubeSpec(
        cube="SWE-bench Verified",
        package_dir="swebench-verified-cube",
        provenance="image",
        task_metadata="src/swebench_verified_cube/task_metadata.json",
        image_evidence="swebench/sweb.eval.x86_64.*",
    ),
    CubeSpec(
        cube="SWE-bench Live",
        package_dir="swebench-live-cube",
        provenance="image",
        task_metadata="src/swebench_live_cube/task_metadata.json",
        image_evidence="starryzhang/sweb.eval.x86_64.*",
    ),
    CubeSpec(
        cube="Terminal-Bench",
        package_dir="terminalbench2-cube",
        provenance="image",
        task_metadata="src/terminalbench2_cube/task_metadata.json",
        image_evidence="upstream per-task image",
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# AST-level function comparison
# ─────────────────────────────────────────────────────────────────────────────
_LOG_ROOTS = frozenset({"logger", "logging", "print", "log", "_logger"})


class _SemanticNormaliser(ast.NodeTransformer):
    """Strip edits that cannot change a score: docstrings, logging, annotations.

    Wrapping a benchmark inevitably reflows docstrings, drops the upstream
    logging calls that would spam a Ray worker's stdout, and modernises type
    annotations under ``ruff``. None of that can change what ``evaluate()``
    returns, so comparing *scoring behaviour* means normalising it away first.
    Everything that survives this pass is a candidate for genuine divergence
    and is enumerated in the report rather than silently absorbed.
    """

    _BODY_FIELDS = ("body", "orelse", "finalbody")

    @staticmethod
    def _is_logging(stmt: ast.stmt) -> bool:
        """True for a bare ``logger.info(...)`` / ``print(...)`` statement."""
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            return False
        root: ast.expr = stmt.value.func
        while isinstance(root, ast.Attribute):
            root = root.value
        return isinstance(root, ast.Name) and root.id in _LOG_ROOTS

    @staticmethod
    def _is_docstring(stmt: ast.stmt) -> bool:
        return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        """Strip docstrings and logging from every statement list in the tree."""
        node = super().generic_visit(node)
        is_scope = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module)
        for fieldname in self._BODY_FIELDS:
            body = getattr(node, fieldname, None)
            if not isinstance(body, list) or not body or not isinstance(body[0], ast.stmt):
                continue
            kept = [
                stmt
                for i, stmt in enumerate(body)
                if not self._is_logging(stmt)
                and not (is_scope and fieldname == "body" and i == 0 and self._is_docstring(stmt))
            ]
            setattr(node, fieldname, kept or ([ast.Pass()] if fieldname != "orelse" else []))
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Drop annotations and decorators: neither can change a returned score."""
        node.returns = None
        node.decorator_list = []
        for arg in [*node.args.args, *node.args.kwonlyargs, *node.args.posonlyargs]:
            arg.annotation = None
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        """``except E as e`` ≡ ``except E`` when the binding is never read.

        A bare ``except:`` is also normalised to ``except Exception:``: the
        ``ruff`` E722 fix rewrites one into the other and no evaluator in the
        corpus relies on catching ``BaseException``.
        """
        if node.type is None:
            node.type = ast.Name(id="Exception", ctx=ast.Load())
        if node.name and not any(
            isinstance(n, ast.Name) and n.id == node.name for n in ast.walk(ast.Module(body=node.body, type_ignores=[]))
        ):
            node.name = None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        """``x: T = v`` → ``x = v``; a bare ``x: T`` declaration disappears."""
        visited = self.generic_visit(node)
        if not isinstance(visited, ast.AnnAssign) or visited.value is None:
            return ast.Pass()
        return ast.Assign(targets=[visited.target], value=visited.value)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        """``type(x) == T`` ≡ ``isinstance(x, T)`` for the equality lint rewrite."""
        visited = self.generic_visit(node)
        if not isinstance(visited, ast.Compare):
            return visited
        left, ops, comparators = visited.left, visited.ops, visited.comparators
        is_type_call = (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Name)
            and left.func.id == "type"
            and len(left.args) == 1
        )
        if is_type_call and len(ops) == 1 and isinstance(ops[0], ast.Eq | ast.NotEq):
            call = ast.Call(
                func=ast.Name(id="isinstance", ctx=ast.Load()),
                args=[left.args[0], comparators[0]],
                keywords=[],
            )
            return ast.UnaryOp(op=ast.Not(), operand=call) if isinstance(ops[0], ast.NotEq) else call
        return visited


def _dump(node: ast.AST) -> str:
    """Position-free AST dump: quote style, wrapping and comments all vanish."""
    return ast.dump(node, annotate_fields=True, include_attributes=False)


class _AlphaRenamer(ast.NodeTransformer):
    """Rename local bindings to positional slots, making renames invisible.

    Lint rules like ``ruff`` E741 force renames such as ``l`` → ``row``. Those
    cannot change a score, but they defeat a structural comparison. Renaming
    every local binding to ``v0, v1, …`` in order of first binding makes two
    alpha-equivalent functions compare equal while still distinguishing
    functions that genuinely use different variables in different places.
    Attributes, globals, and keyword argument names are untouched.
    """

    def __init__(self, locals_: list[str]) -> None:
        self._map = {name: f"v{i}" for i, name in enumerate(locals_)}

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self._map:
            node.id = self._map[node.id]
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg in self._map:
            node.arg = self._map[node.arg]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        if node.name and node.name in self._map:
            node.name = self._map[node.name]
        return self.generic_visit(node)


def _local_bindings(node: ast.AST) -> list[str]:
    """Names bound inside a function, in order of first binding."""
    order: list[str] = []

    def note(name: str) -> None:
        if name not in order:
            order.append(name)

    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            note(arg.arg)
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
            note(sub.id)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            note(sub.name)
    return order


def _normalised_node(node: ast.AST) -> ast.AST:
    """A copy of ``node`` with score-irrelevant edits stripped."""
    stripped = ast.fix_missing_locations(_SemanticNormaliser().visit(copy.deepcopy(node)))
    renamed = _AlphaRenamer(_local_bindings(stripped)).visit(stripped)
    return ast.fix_missing_locations(renamed)


def _semantic_dump(node: ast.AST) -> str:
    """AST dump after stripping score-irrelevant edits (see _SemanticNormaliser)."""
    return _dump(_normalised_node(node))


def _semantic_source(node: ast.AST) -> list[str]:
    """Canonical source lines for the normalised form, for readable diffing."""
    return ast.unparse(_normalised_node(node)).splitlines()


def _functions(path: Path) -> dict[str, tuple[str, str, ast.AST]]:
    """Map function name → (verbatim dump, semantic dump, node) for one module."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return {}
    out: dict[str, tuple[str, str, ast.AST]] = {}

    def add(name: str, node: ast.AST) -> None:
        out[name] = (_dump(node), _semantic_dump(node), node)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            add(node.name, node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef):
                    add(f"{node.name}.{sub.name}", sub)
    return out


def _imports(path: Path) -> set[str]:
    """Top-level imported module names for one module."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def classify_divergence(diff: str) -> str:
    """Bucket one residual diff into a reviewable category.

    Vendoring an evaluator into a wrapper produces a few recurring, benign
    edit shapes. Naming them keeps the headline count honest: a single
    mechanical refactor applied to ninety functions is not ninety independent
    deviations from the original benchmark.

    ``helper_refactor``    the per-OS path-resolution preamble upstream repeats
                           in every getter is replaced by a call to one
                           module-private helper.
    ``platform_pruning``   branches for operating systems the CUBE never runs
                           (a Windows-only or Linux-only VM image) are dropped.
    ``transport``          the VM/file access path changes to the CUBE's
                           controller, leaving the comparison logic intact.
    ``behaviour_change``   anything else: enumerated individually in the report.
    """
    removed = [ln[1:] for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    added = [ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    removed_text, added_text = "\n".join(removed), "\n".join(added)

    os_markers = ("vm_platform", "os_type", "platform.machine", "Darwin", "Linux", "LOCALAPPDATA")
    transport_markers = ("controller.", "execute_python_command", "get_file(", "relocate", "requests.get")
    helper_call = re.compile(r"=\s*_[a-z][a-z0-9_]*\(")

    touches_os = any(m in removed_text for m in os_markers)
    if touches_os and helper_call.search(added_text):
        return "helper_refactor"
    if touches_os:
        return "platform_pruning"
    if any(m in removed_text or m in added_text for m in transport_markers) and len(added) <= 8:
        return "transport"
    if len(added) + len(removed) <= 4:
        return "minor_edit"
    return "behaviour_change"


@dataclass
class VendorReport:
    """AST-comparison result for one vendored directory pair.

    Two identity levels are tracked. *verbatim* means the function's AST is
    byte-for-byte the upstream one (only whitespace/quote/comment edits).
    *semantic* additionally normalises docstrings, logging calls and type
    annotations, which cannot change a score. Functions that differ even
    semantically are listed in ``residual_diffs`` for manual classification.
    """

    files_compared: int = 0
    functions_upstream: int = 0
    functions_verbatim: int = 0
    functions_semantic: int = 0
    functions_divergent: list[str] = field(default_factory=list)
    functions_removed: list[str] = field(default_factory=list)
    functions_added: list[str] = field(default_factory=list)
    imports_dropped: list[str] = field(default_factory=list)
    residual_diffs: dict[str, str] = field(default_factory=dict)
    divergence_classes: dict[str, str] = field(default_factory=dict)

    @property
    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cls in self.divergence_classes.values():
            counts[cls] = counts.get(cls, 0) + 1
        return counts

    def _pct(self, n: int) -> float:
        return 100.0 * n / self.functions_upstream if self.functions_upstream else 0.0

    @property
    def verbatim_pct(self) -> float:
        return self._pct(self.functions_verbatim)

    @property
    def semantic_pct(self) -> float:
        return self._pct(self.functions_semantic)


def compare_vendored(cube_dir: Path, upstream_dir: Path) -> VendorReport:
    """Compare every ``*.py`` in a vendored directory against upstream."""
    rep = VendorReport()
    for up_file in sorted(upstream_dir.glob("*.py")):
        cube_file = cube_dir / up_file.name
        if not cube_file.exists():
            continue
        rep.files_compared += 1
        up_fns, cube_fns = _functions(up_file), _functions(cube_file)
        rep.functions_upstream += len(up_fns)
        for name, (verbatim, semantic, node) in up_fns.items():
            if name not in cube_fns:
                rep.functions_removed.append(f"{up_file.name}:{name}")
                continue
            cube_verbatim, cube_semantic, cube_node = cube_fns[name]
            if cube_verbatim == verbatim:
                rep.functions_verbatim += 1
                rep.functions_semantic += 1
            elif cube_semantic == semantic:
                rep.functions_semantic += 1
            else:
                key = f"{up_file.name}:{name}"
                rep.functions_divergent.append(key)
                diff_text = "\n".join(
                    difflib.unified_diff(
                        _semantic_source(node),
                        _semantic_source(cube_node),
                        fromfile="upstream",
                        tofile="cube",
                        lineterm="",
                        n=1,
                    )
                )
                rep.residual_diffs[key] = diff_text
                rep.divergence_classes[key] = classify_divergence(diff_text)
        rep.functions_added.extend(f"{up_file.name}:{n}" for n in cube_fns if n not in up_fns)
        dropped = _imports(up_file) - _imports(cube_file)
        rep.imports_dropped.extend(f"{up_file.name}:{m}" for m in sorted(dropped))
    return rep


# ─────────────────────────────────────────────────────────────────────────────
# Task-set identity
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TaskSetReport:
    """Task-id and instruction-string comparison against the upstream index."""

    cube_tasks: int = 0
    upstream_tasks: int = 0
    ids_matched: int = 0
    ids_only_in_cube: list[str] = field(default_factory=list)
    ids_only_upstream: list[str] = field(default_factory=list)
    text_compared: int = 0
    text_identical: int = 0

    @property
    def text_identical_pct(self) -> float:
        if not self.text_compared:
            return 0.0
        return 100.0 * self.text_identical / self.text_compared


def compare_task_sets(spec: CubeSpec, cube_root: Path, upstream_root: Path | None) -> TaskSetReport:
    """Compare the CUBE's shipped task list against the upstream task index."""
    rep = TaskSetReport()
    meta_path = cube_root / (spec.task_metadata or "")
    if not spec.task_metadata or not meta_path.exists():
        return rep
    cube_meta = json.loads(meta_path.read_text())
    cube_by_id = {e[spec.task_id_key]: e for e in cube_meta}
    rep.cube_tasks = len(cube_by_id)

    if upstream_root is None or not spec.upstream_tasks_glob:
        return rep

    up_by_id: dict[str, dict[str, Any]] = {}
    for path in upstream_root.glob(spec.upstream_tasks_glob):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and spec.task_id_key in data:
            up_by_id[data[spec.task_id_key]] = data
    rep.upstream_tasks = len(up_by_id)

    matched = set(cube_by_id) & set(up_by_id)
    rep.ids_matched = len(matched)
    rep.ids_only_in_cube = sorted(set(cube_by_id) - set(up_by_id))[:20]
    rep.ids_only_upstream = sorted(set(up_by_id) - set(cube_by_id))[:20]

    if spec.task_text_key:
        for tid in sorted(matched):
            up_text = up_by_id[tid].get(spec.task_text_key)
            cube_text = cube_by_id[tid].get(spec.cube_text_key)
            if up_text is None or cube_text is None:
                continue
            rep.text_compared += 1
            if str(up_text).strip() == str(cube_text).strip():
                rep.text_identical += 1
    return rep


# ─────────────────────────────────────────────────────────────────────────────
# Provenance evidence
# ─────────────────────────────────────────────────────────────────────────────
def declared_dependency(cube_root: Path, package: str) -> str | None:
    """Return the raw dependency line declaring ``package`` in pyproject.toml."""
    pyproject = cube_root / "pyproject.toml"
    if not pyproject.exists():
        return None
    for line in pyproject.read_text().splitlines():
        stripped = line.strip().strip(",").strip('"')
        if stripped.split(">")[0].split("[")[0].split("=")[0].strip() == package:
            return stripped
    return None


def clone_upstream(repo: UpstreamRepo, dest: Path) -> None:
    """Shallow-clone (blobless) an upstream repo and pin it to the audit commit."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", "--filter=blob:none", repo.url, str(dest)],
        check=True,
    )
    if repo.commit:
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", repo.commit], check=True)


def upstream_head(dest: Path) -> str:
    """Short commit hash of a cloned upstream repo."""
    out = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def latex_table(rows: list[dict[str, Any]]) -> str:
    """Render the audit as the LaTeX table used in the paper appendix."""
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        (
            r"\textbf{CUBE} & \textbf{Evaluation provenance} & \textbf{Tasks} "
            r"& \textbf{Task-set match} & \textbf{Evaluator match} \\"
        ),
        r"\midrule",
    ]
    for r in rows:
        task_match = (
            "---"
            if r["task_set"]["upstream_tasks"] == 0
            else f"{r['task_set']['ids_matched']}/{r['task_set']['upstream_tasks']}"
        )
        if r["provenance"] == "vendored":
            v = r["vendored"]
            eval_match = f"{v['functions_semantic']}/{v['functions_upstream']} fn ({v['semantic_pct']:.1f}\\%)"
        elif r["provenance"] == "package":
            eval_match = r"upstream pkg \texttt{" + (r.get("upstream_package") or "") + "}"
        else:
            eval_match = "upstream image + tests"
        lines.append(
            f"{r['cube']} & {r['provenance']} & {r['task_set']['cube_tasks'] or '---'} & {task_match} & {eval_match} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main(
    upstream_dir: Annotated[Path | None, typer.Option(help="Directory holding upstream clones.")] = None,
    clone: Annotated[bool, typer.Option(help="Clone missing upstream repos first.")] = False,
    out: Annotated[Path, typer.Option(help="Where to write the JSON report.")] = Path("parity_audit.json"),
) -> None:
    """Audit every CUBE in the paper corpus against its upstream implementation."""
    upstream_dir = upstream_dir or Path.home() / ".cache" / "cube-parity-upstream"
    rows: list[dict[str, Any]] = []

    for spec in SPECS:
        cube_root = CUBES / spec.package_dir
        if not cube_root.exists():
            typer.echo(f"skip {spec.cube}: {cube_root} missing")
            continue

        repo_dirs: dict[str, Path] = {}
        for key, repo in (("primary", spec.repo), *spec.extra_repos):
            if repo is None:
                continue
            path = upstream_dir / (spec.package_dir if key == "primary" else key)
            if clone:
                clone_upstream(repo, path)
            if path.exists():
                repo_dirs[key] = path
            else:
                typer.echo(f"warn {spec.cube}: no upstream clone at {path} (pass --clone)")
        repo_dir = repo_dirs.get("primary")

        vendored = VendorReport()
        for cube_rel, up_rel, repo_key in spec.vendored_pairs:
            source = repo_dirs.get(repo_key)
            if source is not None:
                part = compare_vendored(cube_root / cube_rel, source / up_rel)
                vendored.files_compared += part.files_compared
                vendored.functions_upstream += part.functions_upstream
                vendored.functions_verbatim += part.functions_verbatim
                vendored.functions_semantic += part.functions_semantic
                vendored.functions_divergent += part.functions_divergent
                vendored.functions_removed += part.functions_removed
                vendored.functions_added += part.functions_added
                vendored.imports_dropped += part.imports_dropped
                vendored.residual_diffs.update(part.residual_diffs)
                vendored.divergence_classes.update(part.divergence_classes)

        task_set = compare_task_sets(spec, cube_root, repo_dir)

        row: dict[str, Any] = {
            "cube": spec.cube,
            "provenance": spec.provenance,
            "upstream_repo": spec.repo.url if spec.repo else None,
            "upstream_commit": upstream_head(repo_dir) if repo_dir else None,
            "upstream_package": spec.upstream_package,
            "declared_dependency": (
                declared_dependency(cube_root, spec.upstream_package) if spec.upstream_package else None
            ),
            "image_evidence": spec.image_evidence,
            "vendored": {
                **{k: v for k, v in asdict(vendored).items() if k != "residual_diffs"},
                "verbatim_pct": vendored.verbatim_pct,
                "semantic_pct": vendored.semantic_pct,
                "class_counts": vendored.class_counts,
            },
            "task_set": {**asdict(task_set), "text_identical_pct": task_set.text_identical_pct},
        }
        rows.append(row)

        typer.echo(f"\n── {spec.cube} [{spec.provenance}]")
        if spec.upstream_package:
            typer.echo(f"   upstream package dependency: {row['declared_dependency']}")
        if vendored.functions_upstream:
            typer.echo(
                f"   evaluator functions: {vendored.functions_upstream} upstream over "
                f"{vendored.files_compared} files\n"
                f"     verbatim AST match:  {vendored.functions_verbatim} ({vendored.verbatim_pct:.1f}%)\n"
                f"     semantic match:      {vendored.functions_semantic} ({vendored.semantic_pct:.1f}%)\n"
                f"     divergent:           {len(vendored.functions_divergent)}\n"
                f"     absent in cube:      {len(vendored.functions_removed)}"
            )
            if vendored.functions_divergent:
                classes = ", ".join(f"{k}={v}" for k, v in sorted(vendored.class_counts.items()))
                typer.echo(f"   divergence classes: {classes}")
                behaviour = [k for k, v in sorted(vendored.divergence_classes.items()) if v == "behaviour_change"]
                if behaviour:
                    typer.echo(f"   behaviour changes ({len(behaviour)}): {', '.join(behaviour[:10])}")
            if vendored.functions_removed:
                typer.echo(f"   absent:    {', '.join(vendored.functions_removed[:8])}")
            diff_path = out.with_name(f"{spec.package_dir}-residual-diffs.txt")
            diff_path.write_text("\n\n".join(f"=== {k}\n{v}" for k, v in sorted(vendored.residual_diffs.items())))
        if task_set.cube_tasks:
            typer.echo(
                f"   task set: {task_set.cube_tasks} shipped, {task_set.upstream_tasks} upstream, "
                f"{task_set.ids_matched} ids matched, "
                f"{task_set.text_identical}/{task_set.text_compared} instruction strings byte-identical"
            )

    out.write_text(json.dumps(rows, indent=2))
    typer.echo(f"\nwrote {out}\n")
    typer.echo(latex_table(rows))


if __name__ == "__main__":
    typer.run(main)
