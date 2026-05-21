# cube-harness Review Rules

This document contains enforceable review rules derived from the [cube-harness Constitution](./constitution.md).

Each rule maps to a specific directive in the constitution and includes severity levels and code examples for automated review.

---

## Severity Levels

All rules are **advisory** - they provide feedback without blocking merges.

| Severity | Description |
|----------|-------------|
| **WARNING** | Should be addressed. Author acknowledgment expected. |
| **SUGGESTION** | Optional improvement. Nice to have. |

---

## Review Output Format

Reviews follow this structure:

1. **Summary** - Brief assessment of changes
2. **Warnings** - Issues that should be addressed (with file:line references)
3. **Suggestions** - Optional improvements
4. **Constitution Compliance** - Pass/Needs Attention for each pillar

---

## Pillar I: Team Contracts & Ownership

### TC-001: RFC for Breaking Changes

**Severity**: WARNING

Breaking changes to the Core API require a Request For Comments (RFC).

**Check**: If a PR modifies public API signatures in core modules (`agent.py`, `environment.py`, `tool.py`, `benchmark.py`, `core.py`), verify there's an associated RFC document or issue reference.

**Applies to**: Changes that alter method signatures, remove public methods, or change return types in core abstractions.

---

## Pillar II: Explicitness

### EX-001: Module-Level Imports Only

**Severity**: WARNING

All imports must be at the top of the module. No imports inside functions or classes.

**Bad**:

```python
def my_function():
    import json  # VIOLATION
    return json.dumps({})
```

**Good**:

```python
import json

def my_function():
    return json.dumps({})
```

### EX-002: No Global Mutable State

**Severity**: WARNING

No module-level mutable variables (lists, dicts, sets, or class instances that accumulate state). Logger instances are acceptable.

**Bad**:

```python
CACHE = {}  # VIOLATION - mutable global

def get_value(key):
    return CACHE.get(key)
```

**Good**:

```python
class Cache:
    def __init__(self):
        self._data = {}

    def get(self, key):
        return self._data.get(key)
```

### EX-003: Composition Over Inheritance

**Severity**: WARNING

Prefer composition (has-a) over inheritance (is-a). Avoid deep inheritance trees where subclasses inherit many unused methods.

**Bad**:

```python
class MyAgent(BaseAllKnowingAgent):  # Inherits 50+ methods
    pass
```

**Good**:

```python
class MyAgent(Agent):
    def __init__(self, planner: Planner, memory: Memory):
        self.planner = planner
        self.memory = memory
```

---

## Pillar III: Scalable Research

### SR-001: Local-Dev Compatible

**Severity**: WARNING

Features must work in single-process mode for debugging. Agent logic must be testable on a laptop without a cluster.

**Check**: New features should not hard-require Ray or distributed execution. Provide local fallbacks or mocks where feasible.

### SR-002: Trace-First Engineering

**Severity**: WARNING

New operations should have appropriate logging and tracing. The "Trace" (logs, screenshots, tool outputs, reasoning steps) is a first-class data product.

**Check**: Long-running operations should be traceable. Errors should include sufficient context for debugging.

### SR-003: Escape Hatch (Raw Access)

**Severity**: SUGGESTION

Abstractions should expose the raw underlying object when necessary for advanced use cases.

**Good**:

```python
class BrowserTool:
    @property
    def raw(self) -> Page:
        """Access the underlying Playwright Page object."""
        return self._page
```

---

## Pillar IV: Protocol Strategy

### PS-001: Hermetic Reproducibility

**Severity**: WARNING

Experiments must capture sufficient information for reproducibility:

- The exact git commit hash
- The full Configuration object (dumped as YAML/JSON)
- The Docker container ID/hash of the environment (if applicable)

### PS-002: Embrace Standards

**Severity**: WARNING

Use established standards instead of inventing new ones:

- **LLM calls**: Use LiteLLM abstractions (not direct OpenAI/Anthropic SDK calls)
- **Tools**: Support Model Context Protocol (MCP) where applicable
- **Data**: Use Agent Data Protocol (ADP) format for traces

**Bad**:

```python
import openai
response = openai.chat.completions.create(...)  # Direct SDK call
```

**Good**:

```python
from cube_harness.llm import LLM
llm = LLM(config)
response = llm.call(prompt)  # Uses LiteLLM internally
```

---

## Pillar V: Code Craft

### CC-001: Type Hints Required

**Severity**: WARNING

All functions must have type hints for parameters and return values, including test functions.

**Bad**:

```python
def process(data):  # Missing types
    return data
```

**Good**:

```python
def process(data: dict) -> dict:
    return data
```

### CC-002: Testing Pyramid

**Severity**: WARNING

New features should have unit tests. The core test suite must run fast (< 5 mins). Slow tests go to nightly builds.

**Check**: PRs adding new functionality should include corresponding tests in `tests/`.

### CC-003: No Vibe Coding

**Severity**: WARNING

AI-assisted code is welcome, but every line must be understood and curated. No blindly pasting large blocks of generated code.

**Signs of vibe coding**:

- Verbose, repetitive logic that could be simplified
- Unnecessary abstractions or over-engineering
- Code that doesn't match project patterns
- Unexplained "magic" constants or logic

### CC-004: Function Atomicity

**Severity**: SUGGESTION

Functions should perform one logical operation and fit on a standard screen (~50-80 lines). Prefer named helpers over inline comments explaining long blocks.

**Bad**:

```python
def process_data(data):
    # 100 lines of mixed responsibilities
    ...
```

**Good**:

```python
def process_data(data: dict) -> dict:
    validated = _validate_input(data)
    transformed = _apply_transformations(validated)
    return _format_output(transformed)
```

### CC-005: Minimalist Imperative

**Severity**: SUGGESTION

Prefer a smaller, simpler codebase. If a feature adds significant complexity but is rarely used, reject it. Refactoring to delete code is valuable.

**Check**: Does this PR add complexity that could be avoided? Could existing code be simplified instead of adding new abstractions?

---

## Pillar V: Code Craft (cont.)

### AF-001: Auto-fix marker integrity

**Severity**: WARNING

Auto-CUBE-produced fixes carry `# auto-fix(N)↓ … # /auto-fix(N)` markers and a
one-line machine-readable footnote at module bottom. `N` is the PR number
(L0/L1) or design-debt issue number (L2/L3). Spec:
`openspec/specs/auto-fix/spec.md`.

**Check** (Tier-2 semantic review; Tier-1 structural checks are in
`scripts/auto_fix_lint.py`): when a diff touches code inside an
`auto-fix(N)` region, crosses a marker boundary, or removes/edits a
marker or footnote, treat it as **possibly rotten**:

- Pull the PR or design-debt issue at `N`; verify the fix's stated invariant still holds after the change.
- If the change is unrelated churn → require the footnote `hash=` be
  re-stamped (acknowledgement, not prevention — never silently leave drift).
- If the fix now looks **subsumed** by surrounding code (the real fix
  landed) → recommend promoting the band-aid into permanent code and
  closing the design-debt issue, rather than keeping a dead marker.
- Orphaned marker (no footnote) or orphaned footnote (no marker) → flag.

**Not** a hard block: flag for acknowledgement. Hard-failing makes
auto-fix blocks radioactive and invites marker deletion.

---

## Quick Reference

| Rule ID | Pillar | Severity | Summary |
|---------|--------|----------|---------|
| TC-001 | Team Contracts | WARNING | RFC for breaking API changes |
| EX-001 | Explicitness | WARNING | Module-level imports only |
| EX-002 | Explicitness | WARNING | No global mutable state |
| EX-003 | Explicitness | WARNING | Composition over inheritance |
| SR-001 | Scalability | WARNING | Local-dev compatible |
| SR-002 | Scalability | WARNING | Trace-first engineering |
| SR-003 | Scalability | SUGGESTION | Escape hatch for raw access |
| PS-001 | Protocols | WARNING | Hermetic reproducibility |
| PS-002 | Protocols | WARNING | Use LiteLLM, MCP, ADP standards |
| CC-001 | Code Craft | WARNING | Type hints required |
| CC-002 | Code Craft | WARNING | Testing pyramid |
| CC-003 | Code Craft | WARNING | No vibe coding |
| CC-004 | Code Craft | SUGGESTION | Function atomicity |
| CC-005 | Code Craft | SUGGESTION | Minimalist imperative |
| AF-001 | Code Craft | WARNING | Auto-fix marker integrity / rot detection |
