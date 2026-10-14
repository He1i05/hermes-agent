"""T2 capability-manifest routing (UG-2, design §4 C2.4/F2.3/O-A).

The decomposer classifies every child as ARTIFACT (code/files) or BEHAVIOR
(adoption/docs/research/review) BEFORE choosing a profile, then routes:
  phase -> capability -> specific profile.

Profiles declare their capability line in ``profile.yaml``::

    capability: 'ARTIFACT: code, tests, infra'

Missing/undescribed profiles default to ARTIFACT (the historical behavior —
every card used to land on builder-like profiles), so routing stays conservative
for a profile the operator has not tagged yet.

M3 audit: every capability-tag match is recorded as a ``capability_routed``
task_event on the child (no schema change), so the before/after audit can
count mis-routes (F2.1).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import profiles as profiles_mod

logger = logging.getLogger(__name__)

# Bucket headings the decomposer checklist mandates on every child body
# (C2.2/C2.3). m3_audit.py parses the same four names.
BUCKET_HEADINGS = ("Adoption", "Docs", "Research-feedback", "Gate")

_CAPABILITY_RE = re.compile(
    r"(?i)\bcapability\s*:\s*\b(artifact|behavior)\b\s*:(?P<tail>[^\n]*)"
)

# Non-code capability keywords -> the profile that owns that behavior.
# A research/design/feedback card goes to scout; a docs card to wiki-keeper.
_BEHAVIOR_PROFILE = {
    "docs": "wiki-keeper",
    "wiki": "wiki-keeper",
    "knowledge": "wiki-keeper",
    "research": "scout",
    "verification": "scout",
    "review": "scout",
    "design": "scout",
    "adoption": None,  # behavioral change — route by roster, not by keyword
}


@dataclass
class _Capability:
    """Parsed profile capability line."""

    name: str
    tag: str  # "ARTIFACT" | "BEHAVIOR"
    keywords: tuple[str, ...] = ()
    declared: bool = False  # False = profile has no capability line yet


def parse_capability(description: str) -> _Capability:
    """Parse the ARTIFACT/BEHAVIOR tag + keyword list out of a profile's
    description. Tolerant: an undescribed profile parses as an undeclared
    ARTIFACT profile (conservative default)."""
    desc = description or ""
    m = _CAPABILITY_RE.search(desc)
    if not m:
        return _Capability(name="", tag="ARTIFACT", declared=False)
    tag = m.group(1).upper()
    tail = m.group("tail") or ""
    keywords = tuple(
        k for k in (w.strip(" ,;.").lower() for w in re.split(r"[,\n]", tail.split(".", 1)[0]))
        if k
    )
    return _Capability(name="", tag=tag, keywords=keywords, declared=True)


def _build_capability_table() -> dict[str, _Capability]:
    """name -> parsed capability for every profile on the roster."""
    try:
        all_profiles = profiles_mod.list_profiles()
    except Exception as exc:  # never raise from routing
        logger.warning("decompose: capability table build failed: %s", exc)
        return {}
    table = {}
    for p in all_profiles:
        cap = parse_capability(p.description or "")
        cap.name = p.name
        table[p.name] = cap
    return table


# Imperative code verbs: presence in the child's title wins over incidental
# behavior-keyword mentions ("implement the wiki index engine" is ARTIFACT).
_CODE_VERB_RE = re.compile(
    r"(?i)\b(implement|refactor|code|write tests?|add tests?|ship|deploy|"
    r"patch|hotfix|configure|install|migrate|benchmark|optimize|"
    r"build|fix|debug|unit tests?|regression tests?)\b",
)


def classify_child(title: str, body: str) -> str:
    """Infer the child's capability tag from its own title/body when the LLM
    did not emit one. Code-verb-shaped titles are ARTIFACT even when they
    mention a behavior keyword in passing; otherwise docs/research/review
    keyword hits make it BEHAVIOR; else ARTIFACT (conservative)."""
    title_l = (title or "").lower()
    text = f"{title}\n{body or ''}".lower()
    if _CODE_VERB_RE.search(title_l):
        return "ARTIFACT"
    for kw, profile in _BEHAVIOR_PROFILE.items():
        if profile is not None and re.search(rf"(?i)\b{re.escape(kw)}\b", text):
            return "BEHAVIOR"
    if re.search(r"(?i)\badoption\b|behavioral change", text):
        return "BEHAVIOR"
    return "ARTIFACT"


def route_assignee(
    tag: str, declared_assignee: Optional[str], routing: "_RoutingLike",
    caps: dict[str, _Capability], *, hint_text: str = "",
) -> tuple[str, str]:
    """C2.4 routing decision: phase first (already in the LLM's choice), then
    capability, then specific profile. Returns ``(assignee, reason)``:

    - ``llm``        LLM pick honored (matching capability, or the chosen
                     profile has no declared capability line — routing stays
                     conservative and the existing fallback path is untouched);
    - ``capability:<profile>``  declared capability mismatch, rerouted to the
                     declared profile carrying the child's tag (unique tag
                     owner, or unique keyword hit inside ``hint_text``);
    - ``default``    no/invalid LLM pick and no declared profile matches the
                     tag — default_assignee fallback intact."""
    tag = (tag or "").upper()
    if tag not in ("ARTIFACT", "BEHAVIOR"):
        tag = "ARTIFACT"
    if not caps:
        # No capability data at all (build failed / empty roster): preserve the
        # LLM pick verbatim — legacy fallback routing intact.
        return (declared_assignee or routing.default_assignee,
                "llm" if declared_assignee else "default")

    def _keyword_pick(candidates: list[str]) -> Optional[str]:
        """Unique candidate whose declared keywords appear in hint_text."""
        if not hint_text:
            return None
        low = hint_text.lower()
        scored = [
            (n, sum(1 for kw in caps[n].keywords if kw and kw in low))
            for n in candidates
        ]
        scored = [(n, s) for n, s in scored if s > 0]
        if len(scored) == 1 or (
            scored and len({s for _, s in scored}) > 1
            and max(scored, key=lambda p: p[1])[1] > sorted((s for _, s in scored), reverse=True)[1]
        ):
            best = max(scored, key=lambda p: p[1])
            # Unique strict leader only.
            if sum(1 for _, s in scored if s == best[1]) == 1:
                return best[0]
        return None

    declared_any = any(c.declared for c in caps.values())
    if declared_assignee and declared_assignee in caps:
        cap = caps[declared_assignee]
        if not cap.declared or cap.tag == tag:
            return declared_assignee, "llm"
        # Declared capability mismatch (F2.1): reroute to the declared profile
        # carrying the child's tag, when exactly one exists; with several,
        # fall back to a keyword hit in the child's own text.
        candidates = [n for n, c in caps.items() if c.tag == tag and c.declared]
        if len(candidates) == 1:
            return candidates[0], f"capability:{candidates[0]}"
        pick = _keyword_pick(candidates)
        if pick:
            return pick, f"capability:{pick}"
        logger.info(
            "decompose: capability mismatch %s->%s but %d declared %s profiles; keeping LLM pick",
            declared_assignee, tag, len(candidates), tag.lower(),
        )
        return declared_assignee, "llm_mismatch_kept"
    # No valid LLM pick: route by capability tag when a declared profile owns
    # it. When NO profile on the roster declares a capability line at all,
    # keep the legacy default path untouched (fallback routing intact).
    if not declared_any:
        return routing.default_assignee, "default"
    candidates = [n for n, c in caps.items() if c.tag == tag and c.declared]
    if len(candidates) == 1:
        return candidates[0], f"capability:{candidates[0]}"
    pick = _keyword_pick(candidates)
    if pick:
        return pick, f"capability:{pick}"
    return routing.default_assignee, "default"


class _RoutingLike:
    """Structural duck-type of ``kanban_decompose._Routing`` (default_assignee)."""

    default_assignee: str


def _apply_capability_routing(
    task_id: str, children: list[dict], routing, caps: dict[str, _Capability],
) -> int:
    """Enforce the capability-first routing order on the cleaned children list,
    in place. Returns the number of children whose assignee changed."""
    changed = 0
    for idx, child in enumerate(children):
        hint = f"{child['title']} {child.get('body') or ''}"
        tag = child.get("capability_tag") or classify_child(child["title"], child["body"])
        child["capability_tag"] = tag
        chosen, reason = route_assignee(
            tag, child.get("assignee"), routing, caps, hint_text=hint,
        )
        if chosen != child["assignee"]:
            logger.info(
                "decompose: task %s child %d rerouted by capability %s -> %r (%s)",
                task_id, idx, tag, chosen, reason,
            )
            changed += 1
        child["assignee"] = chosen
        child["capability_reason"] = reason
    return changed


def audit_log_routed_children(
    conn, task_id: str, child_ids: list[str], children: list[dict], run_id: Optional[int] = None,
) -> None:
    """M3 audit logging: one ``capability_routed`` event per child (task_events,
    no new column). Payload carries the tag, final assignee and reason so the
    before/after audit can count mis-routes (F2.1) without re-parsing prose."""
    from hermes_cli.kanban_db import _append_event

    try:
        for cid, child in zip(child_ids, children):
            _append_event(
                conn, cid, "capability_routed",
                {
                    "capability_tag": child.get("capability_tag"),
                    "assignee": child.get("assignee"),
                    "reason": child.get("capability_reason"),
                    "root": task_id,
                },
                run_id=run_id,
            )
    except Exception:
        # Audit logging must never fail a fan-out that already committed.
        logger.exception("decompose: M3 capability_routed audit logging failed on task %s", task_id)


def append_bucket_headings(children: list[dict]) -> None:
    """C2.3: every child body carries the 4 bucket headings (Adoption / Docs /
    Research-feedback / Gate), 'none' when empty, so a reviewer can see they
    were considered and m3_audit.py can verify coverage."""
    for child in children:
        body = child.get("body") or ""
        existing = {h.lower() for h in BUCKET_HEADINGS if re.search(
            rf"(?im)^\s*(?:#{{1,4}}\s*|\*\*)?\s*{re.escape(h)}\b", body,
        )}
        lines = []
        for h in BUCKET_HEADINGS:
            if h.lower() not in existing:
                lines.append(f"{h}: none")
        if lines:
            block = "\n\n".join([body.rstrip(), "\n".join(lines)]) if body else "\n".join(lines)
            child["body"] = block