#!/usr/bin/env python3
"""judge_validation.py — human ground-truth validation of the trajectory judge.

Reviewers asked for the one thing agreement-between-judges cannot show: whether
the judge is *correct*, not merely *consistent*. This tool runs that study
end-to-end.

    1. ``sample``  draw a stratified sample of episodes across modalities and
                   models, balanced over the judge's own blame categories so the
                   study can detect per-category error rather than only the
                   accuracy on whatever category happens to dominate.
    2. ``packet``  render a blind annotation instrument: one self-contained HTML
                   file per annotator holding the same transcript the judge read,
                   with the judge's verdict withheld. Annotators label outcome and
                   primary blame from the identical closed taxonomy and export
                   their answers as JSON.
    3. ``score``   compute Cohen's kappa (human vs judge), inter-annotator kappa,
                   and per-modality/per-bucket breakdowns, and emit the LaTeX
                   tables for the appendix.

Blinding matters: the packet never contains the judge's label, so an annotator
cannot anchor on it. Stratification is by (modality, judge blame) so that rare
categories -- the benchmark-side ones that carry the faithfulness argument --
get enough samples to estimate agreement at all.

Usage
-----
    scripts/paper/judge_validation.py sample  ~/cube_harness_results --n 100 --out study/
    scripts/paper/judge_validation.py packet  study/ --annotator alice
    scripts/paper/judge_validation.py score   study/ --labels study/labels-*.json
"""

from __future__ import annotations

import html
import json
import random
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from cube_harness.analyze.investigator.episode_discovery import discover_episodes
from cube_harness.analyze.investigator.transcript import extract_transcript
from cube_harness.eval_log import BlameCategory, Outcome

app = typer.Typer(add_completion=False, help=__doc__)

# The three buckets the paper's faithfulness argument rests on. Kappa is
# reported both on the full 10-way taxonomy and on this coarser grouping,
# because a judge that confuses model_capability with agent_scaffolding is
# far less consequential than one that confuses agent-side with benchmark-side.
BUCKETS: dict[str, str] = {
    "model_capability": "agent",
    "agent_scaffolding": "agent",
    "submission_format": "agent",
    "tool_failure": "tool",
    "action_space_limited": "tool",
    "insufficient_observation": "tool",
    "env_failure": "benchmark",
    "task_unclear": "benchmark",
    "eval_brittle": "benchmark",
    "none": "none",
}

SAMPLE_FILENAME = "sample.json"
KEY_FILENAME = "judge_key.json"


@dataclass
class SampledEpisode:
    """One episode drawn into the study, with the judge label kept separate."""

    uid: str  # blind identifier shown to annotators
    experiment: str
    trajectory_id: str
    episode_dir: str
    modality: str
    benchmark: str
    model: str
    task_id: str
    task_description: str
    score: float
    judge_outcome: str
    judge_blame: str
    judge_confidence: int


def _modality_of(benchmark: str) -> str:
    """Map a benchmark name onto the paper's modality grouping."""
    name = benchmark.lower()
    if any(k in name for k in ("workarena", "webarena", "miniwob", "browsercomp")):
        return "Web"
    if any(k in name for k in ("osworld", "windows", "waa")):
        return "CUA"
    if any(k in name for k in ("swebench", "swe-bench", "swegym")):
        return "SWE"
    if "terminal" in name:
        return "Terminal"
    return "Other"


def _collect(results_dir: Path) -> list[SampledEpisode]:
    """Read every investigated episode under ``results_dir``."""
    found: list[SampledEpisode] = []
    experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir()] if results_dir.is_dir() else []
    for exp_dir in sorted(experiment_dirs):
        try:
            refs = discover_episodes(exp_dir)
        except (FileNotFoundError, NotADirectoryError):
            # A results root accumulates aborted runs with no episodes/ at all.
            # Skipping them is right: this is a survey, not a validation pass.
            continue
        for ref in refs:
            record = ref.record
            if record is None or record.findings is None:
                continue
            benchmark = getattr(record, "benchmark_name", None) or exp_dir.name
            model = getattr(record, "model_name", None) or getattr(record, "agent_name", "") or "unknown"
            found.append(
                SampledEpisode(
                    uid="",
                    experiment=exp_dir.name,
                    trajectory_id=ref.trajectory_id,
                    episode_dir=str(ref.episode_dir),
                    modality=_modality_of(str(benchmark)),
                    benchmark=str(benchmark),
                    model=str(model),
                    task_id=record.sample_id,
                    task_description=(record.task_description or "")[:4000],
                    score=record.score,
                    judge_outcome=record.findings.outcome.value,
                    judge_blame=record.findings.primary_blame.value,
                    judge_confidence=record.findings.primary_blame_confidence,
                )
            )
    return found


@app.command()
def sample(
    results_dir: Annotated[Path, typer.Argument(help="Root holding experiment directories.")],
    out: Annotated[Path, typer.Option(help="Study directory to create.")] = Path("judge_study"),
    n: Annotated[int, typer.Option(help="Target sample size.")] = 100,
    seed: Annotated[int, typer.Option(help="Sampling seed.")] = 20260725,
    failures_only: Annotated[bool, typer.Option(help="Restrict to episodes that scored 0.")] = True,
    min_per_cell: Annotated[int, typer.Option(help="Floor per (modality, blame) cell.")] = 3,
) -> None:
    """Draw a stratified, blinded sample and write the study directory."""
    rng = random.Random(seed)
    episodes = _collect(results_dir)
    if failures_only:
        episodes = [e for e in episodes if e.score <= 0.0]
    if not episodes:
        raise typer.BadParameter(f"no investigated episodes found under {results_dir}")

    cells: dict[tuple[str, str], list[SampledEpisode]] = defaultdict(list)
    for ep in episodes:
        cells[(ep.modality, ep.judge_blame)].append(ep)

    # Floor every cell first so rare benchmark-side categories are represented,
    # then fill the remaining budget proportionally to cell size.
    chosen: list[SampledEpisode] = []
    for key in sorted(cells):
        pool = cells[key]
        rng.shuffle(pool)
        chosen.extend(pool[: min(min_per_cell, len(pool))])

    remaining = [e for e in episodes if e not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, n - len(chosen))])
    rng.shuffle(chosen)

    for i, ep in enumerate(chosen):
        ep.uid = f"E{i:03d}"

    out.mkdir(parents=True, exist_ok=True)
    blind_fields = (
        "uid",
        "experiment",
        "trajectory_id",
        "episode_dir",
        "modality",
        "benchmark",
        "model",
        "task_id",
        "task_description",
        "score",
    )
    (out / SAMPLE_FILENAME).write_text(
        json.dumps([{k: getattr(ep, k) for k in blind_fields} for ep in chosen], indent=2)
    )
    (out / KEY_FILENAME).write_text(
        json.dumps(
            {
                ep.uid: {"outcome": ep.judge_outcome, "blame": ep.judge_blame, "confidence": ep.judge_confidence}
                for ep in chosen
            },
            indent=2,
        )
    )

    dist = Counter((e.modality, e.judge_blame) for e in chosen)
    typer.echo(f"sampled {len(chosen)} episodes from {len(episodes)} investigated failures")
    for (modality, blame), count in sorted(dist.items()):
        typer.echo(f"  {modality:8s} {blame:26s} {count}")
    typer.echo(f"\nwrote {out / SAMPLE_FILENAME} and {out / KEY_FILENAME} (keep the key away from annotators)")


def _render_packet(entries: list[dict[str, Any]], transcripts: dict[str, str], annotator: str) -> str:
    """Build the self-contained blind annotation page."""
    blames = [b.value for b in BlameCategory]
    outcomes = [o.value for o in Outcome]
    payload = json.dumps(
        [
            {
                "uid": e["uid"],
                "modality": e["modality"],
                "benchmark": e["benchmark"],
                "task_id": e["task_id"],
                "task": e["task_description"],
                "score": e["score"],
                "transcript": transcripts.get(e["uid"], "(transcript unavailable)"),
            }
            for e in entries
        ]
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>CUBE judge validation — {html.escape(annotator)}</title>
<style>
 body {{ font: 14px/1.5 -apple-system, system-ui, sans-serif; margin: 0; display: flex; height: 100vh; }}
 #list {{ width: 200px; overflow-y: auto; border-right: 1px solid #ccc; padding: 8px; }}
 #list div {{ padding: 4px; cursor: pointer; border-radius: 4px; }}
 #list div.done {{ background: #e6f4ea; }}
 #list div.active {{ outline: 2px solid #1a73e8; }}
 #main {{ flex: 1; overflow-y: auto; padding: 16px; }}
 pre {{ background: #f6f8fa; padding: 12px; overflow-x: auto; max-height: 55vh; white-space: pre-wrap; }}
 .task {{ background: #fffbe6; padding: 12px; border-left: 3px solid #f0c000; }}
 label {{ display: block; margin: 4px 0; }}
 button {{ padding: 8px 16px; margin-right: 8px; }}
</style></head><body>
<div id="list"></div>
<div id="main">
  <h2 id="title"></h2>
  <div class="task" id="task"></div>
  <h3>Trajectory</h3>
  <pre id="transcript"></pre>
  <h3>Your assessment</h3>
  <p>Judge the episode on the evidence in the transcript alone. Choose
     <b>none</b> for blame if the transcript does not support any attribution.</p>
  <div id="outcome"></div>
  <div id="blame"></div>
  <label>Notes (optional)<br><textarea id="notes" rows="3" cols="80"></textarea></label>
  <button onclick="save()">Save &amp; next</button>
  <button onclick="exportAll()">Export JSON</button>
  <span id="status"></span>
</div>
<script>
const DATA = {payload};
const OUTCOMES = {json.dumps(outcomes)};
const BLAMES = {json.dumps(blames)};
const ANNOTATOR = {json.dumps(annotator)};
const KEY = 'cube-judge-' + ANNOTATOR;
let labels = JSON.parse(localStorage.getItem(KEY) || '{{}}');
let idx = 0;

function renderList() {{
  document.getElementById('list').innerHTML = DATA.map((d, i) =>
    `<div class="${{labels[d.uid] ? 'done' : ''}} ${{i === idx ? 'active' : ''}}" onclick="go(${{i}})">${{d.uid}} · ${{d.modality}}</div>`).join('');
}}
function render() {{
  const d = DATA[idx];
  document.getElementById('title').textContent = `${{d.uid}} — ${{d.benchmark}} (${{d.modality}}) — final score ${{d.score}}`;
  document.getElementById('task').textContent = d.task || '(no task description recorded)';
  document.getElementById('transcript').textContent = d.transcript;
  const prev = labels[d.uid] || {{}};
  document.getElementById('outcome').innerHTML = '<b>Outcome</b>' + OUTCOMES.map(o =>
    `<label><input type="radio" name="outcome" value="${{o}}" ${{prev.outcome === o ? 'checked' : ''}}> ${{o}}</label>`).join('');
  document.getElementById('blame').innerHTML = '<b>Primary blame</b>' + BLAMES.map(b =>
    `<label><input type="radio" name="blame" value="${{b}}" ${{prev.blame === b ? 'checked' : ''}}> ${{b}}</label>`).join('');
  document.getElementById('notes').value = prev.notes || '';
  renderList();
}}
function go(i) {{ idx = i; render(); window.scrollTo(0, 0); }}
function save() {{
  const d = DATA[idx];
  const outcome = document.querySelector('input[name=outcome]:checked');
  const blame = document.querySelector('input[name=blame]:checked');
  if (!outcome || !blame) {{ alert('Pick an outcome and a primary blame.'); return; }}
  labels[d.uid] = {{ outcome: outcome.value, blame: blame.value, notes: document.getElementById('notes').value }};
  localStorage.setItem(KEY, JSON.stringify(labels));
  document.getElementById('status').textContent = Object.keys(labels).length + '/' + DATA.length + ' labelled';
  if (idx < DATA.length - 1) go(idx + 1); else render();
}}
function exportAll() {{
  const blob = new Blob([JSON.stringify({{annotator: ANNOTATOR, labels}}, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'labels-' + ANNOTATOR + '.json';
  a.click();
}}
render();
</script></body></html>
"""


@app.command()
def packet(
    study_dir: Annotated[Path, typer.Argument(help="Study directory created by `sample`.")],
    annotator: Annotated[str, typer.Option(help="Annotator name (namespaces local storage).")] = "annotator",
    max_chars: Annotated[int, typer.Option(help="Transcript truncation budget per episode.")] = 60000,
) -> None:
    """Render a blind, self-contained HTML annotation packet."""
    entries = json.loads((study_dir / SAMPLE_FILENAME).read_text())
    transcripts: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for entry in entries:
            episode_dir = Path(entry["episode_dir"])
            try:
                out_dir = extract_transcript(episode_dir, Path(tmp) / entry["uid"])
                text = (out_dir / "transcript.txt").read_text(errors="replace")
            except Exception as exc:  # noqa: BLE001 — one corrupt episode must not sink the packet
                text = f"(could not read transcript: {exc})"
            if len(text) > max_chars:
                head, tail = text[: max_chars // 2], text[-max_chars // 2 :]
                text = f"{head}\n\n[... {len(text) - max_chars} characters elided ...]\n\n{tail}"
            transcripts[entry["uid"]] = text

    dest = study_dir / f"packet-{annotator}.html"
    dest.write_text(_render_packet(entries, transcripts, annotator))
    typer.echo(f"wrote {dest} ({len(entries)} episodes, judge labels withheld)")


@app.command()
def snapshot(
    results_dir: Annotated[Path, typer.Argument(help="Root holding experiment directories.")],
    out: Annotated[Path, typer.Option(help="Where to write the snapshot.")] = Path("judge_snapshot.json"),
    label: Annotated[str, typer.Option(help="Name for this judging pass, e.g. 'claude' or 'codex'.")] = "pass",
) -> None:
    """Freeze the current judge labels so a later pass can be compared against them.

    A second judging pass overwrites ``findings`` in each episode record, so the
    backbone-swap comparison is only possible if the first pass is captured
    first. Run this, re-judge with ``ch-investigate --driver codex --overwrite``,
    snapshot again, then ``swap``.
    """
    episodes = _collect(results_dir)
    out.write_text(
        json.dumps(
            {
                "label": label,
                "labels": {
                    f"{e.experiment}/{e.trajectory_id}": {
                        "outcome": e.judge_outcome,
                        "blame": e.judge_blame,
                        "confidence": e.judge_confidence,
                        "modality": e.modality,
                        "benchmark": e.benchmark,
                    }
                    for e in episodes
                },
            },
            indent=2,
        )
    )
    typer.echo(f"snapshotted {len(episodes)} judged episodes as '{label}' → {out}")


@app.command()
def swap(
    first: Annotated[Path, typer.Argument(help="Snapshot from the first judging pass.")],
    second: Annotated[Path, typer.Argument(help="Snapshot from the second (different backbone).")],
    out: Annotated[Path, typer.Option(help="Where to write the LaTeX table.")] = Path("judge_swap.tex"),
) -> None:
    """Compare two judging passes that differ only in the model backbone.

    Answers the question a single-vendor judge cannot: is the blame distribution
    a property of the trajectories, or of the model reading them? Reports
    per-episode agreement and, separately, the shift in the aggregate blame
    distribution, since a judge can disagree episode by episode while producing
    the same corpus-level picture (which is the claim the paper's 82\\% rests on).
    """
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    la, lb = a["labels"], b["labels"]
    shared = sorted(set(la) & set(lb))
    if not shared:
        raise typer.BadParameter("the two snapshots share no episodes")

    blame_a = [la[k]["blame"] for k in shared]
    blame_b = [lb[k]["blame"] for k in shared]
    bucket_a = [BUCKETS.get(x, "other") for x in blame_a]
    bucket_b = [BUCKETS.get(x, "other") for x in blame_b]
    kappa_blame, agree_blame = cohens_kappa(blame_a, blame_b)
    kappa_bucket, agree_bucket = cohens_kappa(bucket_a, bucket_b)

    typer.echo(f"n = {len(shared)} episodes judged by both backbones ({a['label']} vs {b['label']})\n")
    typer.echo(f"primary blame  kappa={kappa_blame:.3f}  agreement={agree_blame * 100:.1f}%")
    typer.echo(f"blame bucket   kappa={kappa_bucket:.3f}  agreement={agree_bucket * 100:.1f}%\n")

    typer.echo(f"{'bucket':12s} {a['label']:>10s} {b['label']:>10s}")
    ca, cb = Counter(bucket_a), Counter(bucket_b)
    for bucket in sorted(set(ca) | set(cb)):
        typer.echo(f"{bucket:12s} {ca[bucket] / len(shared) * 100:9.1f}% {cb[bucket] / len(shared) * 100:9.1f}%")

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        f"\\textbf{{Blame bucket}} & \\textbf{{{a['label']} judge}} & \\textbf{{{b['label']} judge}} & \\textbf{{$\\Delta$}} \\\\",
        r"\midrule",
    ]
    for bucket in sorted(set(ca) | set(cb)):
        pa, pb = ca[bucket] / len(shared) * 100, cb[bucket] / len(shared) * 100
        lines.append(f"{bucket} & {pa:.1f}\\% & {pb:.1f}\\% & {pb - pa:+.1f} \\\\")
    lines += [
        r"\midrule",
        (
            f"\\multicolumn{{4}}{{l}}{{\\small $n={len(shared)}$; per-episode "
            f"$\\kappa={kappa_bucket:.2f}$ on buckets, ${kappa_blame:.2f}$ on the 10-way taxonomy}} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    out.write_text("\n".join(lines))
    typer.echo(f"\nwrote {out}")


def cohens_kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    """Cohen's kappa and raw agreement for two equal-length label sequences."""
    if not a:
        return float("nan"), float("nan")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return kappa, observed


def _interpret(kappa: float) -> str:
    """Landis & Koch strength-of-agreement label."""
    for threshold, label in ((0.81, "almost perfect"), (0.61, "substantial"), (0.41, "moderate"), (0.21, "fair")):
        if kappa >= threshold:
            return label
    return "slight"


@app.command()
def score(
    study_dir: Annotated[Path, typer.Argument(help="Study directory.")],
    labels: Annotated[list[Path] | None, typer.Option(help="One labels-<annotator>.json per annotator.")] = None,
    out: Annotated[Path, typer.Option(help="Where to write the LaTeX table.")] = Path("judge_validation.tex"),
) -> None:
    """Compute human-vs-judge and inter-annotator agreement, and emit LaTeX."""
    entries = {e["uid"]: e for e in json.loads((study_dir / SAMPLE_FILENAME).read_text())}
    key = json.loads((study_dir / KEY_FILENAME).read_text())
    annotations = {p.stem.replace("labels-", ""): json.loads(p.read_text())["labels"] for p in labels or []}
    if not annotations:
        raise typer.BadParameter("pass at least one --labels file")

    # Human consensus: majority blame across annotators; ties resolve to the
    # first annotator in sorted order, and are counted so they can be reported.
    names = sorted(annotations)
    consensus: dict[str, dict[str, str]] = {}
    ties = 0
    for uid in entries:
        votes = [annotations[a][uid] for a in names if uid in annotations[a]]
        if not votes:
            continue
        blame_counts = Counter(v["blame"] for v in votes)
        top = blame_counts.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            ties += 1
        consensus[uid] = {
            "blame": top[0][0],
            "outcome": Counter(v["outcome"] for v in votes).most_common(1)[0][0],
        }

    uids = sorted(consensus)
    human_blame = [consensus[u]["blame"] for u in uids]
    judge_blame = [key[u]["blame"] for u in uids]
    human_bucket = [BUCKETS.get(b, "other") for b in human_blame]
    judge_bucket = [BUCKETS.get(b, "other") for b in judge_blame]
    human_outcome = [consensus[u]["outcome"] for u in uids]
    judge_outcome = [key[u]["outcome"] for u in uids]

    rows: list[tuple[str, str, int, float, float]] = []

    def add(label: str, subset: str, gold: list[str], pred: list[str]) -> None:
        kappa, agree = cohens_kappa(gold, pred)
        rows.append((label, subset, len(gold), kappa, agree))

    add("Primary blame (10-way)", "all", human_blame, judge_blame)
    add("Blame bucket (agent/tool/benchmark)", "all", human_bucket, judge_bucket)
    add("Outcome", "all", human_outcome, judge_outcome)

    for modality in sorted({entries[u]["modality"] for u in uids}):
        idx = [i for i, u in enumerate(uids) if entries[u]["modality"] == modality]
        add("Primary blame (10-way)", modality, [human_blame[i] for i in idx], [judge_blame[i] for i in idx])
        add("Blame bucket", modality, [human_bucket[i] for i in idx], [judge_bucket[i] for i in idx])

    typer.echo(f"{len(uids)} episodes scored, {len(names)} annotators ({', '.join(names)}), {ties} consensus ties\n")
    typer.echo(f"{'metric':40s} {'subset':10s} {'n':>4s} {'kappa':>7s} {'agree':>7s}")
    for label, subset, n, kappa, agree in rows:
        typer.echo(f"{label:40s} {subset:10s} {n:4d} {kappa:7.3f} {agree:7.3f}")

    if len(names) > 1:
        typer.echo("\ninter-annotator (pairwise Cohen's kappa on primary blame):")
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                shared = [u for u in entries if u in annotations[a] and u in annotations[b]]
                kappa, agree = cohens_kappa(
                    [annotations[a][u]["blame"] for u in shared], [annotations[b][u]["blame"] for u in shared]
                )
                typer.echo(f"  {a} vs {b}: n={len(shared)} kappa={kappa:.3f} agreement={agree:.3f}")

    lines = [
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        r"\textbf{Agreement} & \textbf{Subset} & $n$ & $\kappa$ & \textbf{Raw} & \textbf{Strength} \\",
        r"\midrule",
    ]
    lines += [
        f"{label} & {subset} & {n} & {kappa:.2f} & {agree * 100:.1f}\\% & {_interpret(kappa)} \\\\"
        for label, subset, n, kappa, agree in rows
    ]
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines))
    typer.echo(f"\nwrote {out}")


if __name__ == "__main__":
    app()
