"""Symlink each `use_cases/<name>/SKILL.md` into `.claude/skills/investigator-<name>`.

Run from the repo root. Idempotent: existing symlinks pointing inside
`use_cases/` are refreshed; correct ones are skipped.

Uses *relative* symlink targets so the `.claude/skills/investigator-*` entries
travel through git unchanged and resolve in any checkout (worktrees included).

This lets Auto-CUBE's skill picker enumerate Investigator recipes the same way
it enumerates any other skill, without duplicating the markdown.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

INVESTIGATOR_USE_CASES_DIR = Path("src/cube_harness/analyze/investigator/use_cases")
SKILLS_DIR = Path(".claude/skills")


def _sync(*, repo_root: Path, dry_run: bool) -> list[Path]:
    """Create / refresh the symlinks. Returns the list of links created or updated."""
    use_cases_dir = (repo_root / INVESTIGATOR_USE_CASES_DIR).resolve()
    if not use_cases_dir.exists():
        raise typer.BadParameter(f"use_cases directory not found: {use_cases_dir}")

    skills_dir = repo_root / SKILLS_DIR
    skills_dir.mkdir(parents=True, exist_ok=True)

    created_or_updated: list[Path] = []
    for sub in sorted(use_cases_dir.iterdir()):
        if not sub.is_dir():
            continue
        skill_md = sub / "SKILL.md"
        if not skill_md.exists():
            continue
        link_path = skills_dir / f"investigator-{sub.name}"
        # Relative target so the symlink survives different checkouts / worktrees:
        # `.claude/skills/investigator-<name>` → `../../src/cube_harness/.../SKILL.md`.
        relative_target = Path(os.path.relpath(skill_md.resolve(), start=link_path.parent.resolve()))
        if link_path.is_symlink() or link_path.exists():
            if link_path.is_symlink() and link_path.readlink() == relative_target:
                continue  # already correct
            if not dry_run:
                link_path.unlink()
        if not dry_run:
            link_path.symlink_to(relative_target)
        created_or_updated.append(link_path)
    return created_or_updated


def main(
    repo_root: Annotated[Path, typer.Option(help="Repo root.")] = Path("."),
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would change, do nothing.")] = False,
) -> None:
    """Sync investigator-skill symlinks. Idempotent."""
    changed = _sync(repo_root=repo_root, dry_run=dry_run)
    if not changed:
        typer.echo("All investigator-skill symlinks are up to date.")
        return
    verb = "Would create/update" if dry_run else "Created/updated"
    typer.echo(f"{verb} {len(changed)} symlink(s):")
    for link in changed:
        typer.echo(f"  {link}")


if __name__ == "__main__":
    typer.run(main)
