"""Declared stream coverage contract (W-8, quest t_bec82259 / card t_e9fb49b3).

The defect this module cures
----------------------------
The decomposer's ONLY view of a root body used to be
``kanban_specify._truncate(task.body, 4000)``. The cut lands mid-section, so a
root body that declares streams (``## STREAMS`` / ``## SCOPE STREAMS``, entries
``S1 <description>``) silently loses every stream past char 3999. Measured on
the real roots (read-only, 2026-09-30, code pin ``26b000cc0f``):

    t_26d8eb94  body 8234 chars  S1@3144 S2@3812 | S3@4383 S4@4923 S5@5510 S6@5945
    t_bec82259  body 8501 chars  R1@~3200 R2@~3650 | R3@~4200 R4@~4720 R5@~5030

Only ``S1``/``S2`` were ever in front of the model; ``S3``–``S6`` were never
seen, so no receipt built from the prompt slice can ever report them. A receipt
built as a body-vs-children *diff* false-passes 4/4 on that incident (keyword
overlap: S3–S6 each borrow S1's child on the generic word "pipeline"). The only
oracle that scores 0/4 is a DECLARATION oracle — which requires a contract the
body author writes and the decomposer honours. That contract is this module.

The contract
------------
1. A root body DECLARES its streams, in a section whose heading contains the
   word ``STREAM``/``STREAMS``, as tagged entries::

       ## SCOPE STREAMS (triager to decompose)
       S1 STRUCTURAL AUDIT — map the dependency chain
       S2 SIGNAL ECONOMICS — entry points, calibration

   (a ``streams: S1, S2`` frontmatter line is accepted as the short form).

2. Every declared stream must be accounted for by exactly one emitted child::

       covers: S3                      <- this child owns stream S3
       Fold-in: S4 — absorbed by the S3 census child

   ``covers:`` names the stream the carrying child OWNS; ``Fold-in:`` names the
   stream the carrying child ABSORBS on behalf of another child's deliverable
   (design rule C-5, spelled with the stream tag so the mapping stays
   machine-readable — see ``Note on Fold-in`` below).

3. A declared stream with neither declaration is a COVERAGE GAP.

Scope
-----
This module is the single source of truth for the CONVENTION: tag parsing, the
declaration vocabulary, the full-body prompt projection, and the gap oracle.
It deliberately mirrors the sibling modules ``hermes_cli/kanban_discovery.py``
(T5 discovery reports), ``hermes_cli/kanban_m2.py`` (T4 research acceptance) and
``hermes_cli/kanban_design.py`` (T3 design contract) so the templates cannot
drift apart.

What it does NOT do (owned elsewhere):
  * The post-decompose RECEIPT (stream -> child table + fold-in accounting,
    commented on the root) is W-7 — ``hermes_cli/kanban_coverage.py``, expected
    to build on the primitives here rather than re-deriving them.
  * The fan-out receipt call and ``gate_stamp_census`` are W-9. The DECLARED
    DELIVERABLE marker/selector is W-10: this module owns the literals
    (:data:`DELIVERABLE_KEY` / :data:`DELIVERS_KEY`), the parser
    (:func:`declared_deliverable`), the canonical line renderer and the prompt
    block; ``kanban_gate.select_deliverable`` owns the selection itself.

Backward compatibility
---------------------
Declaration-driven and truncation-gated. ``prompt_body`` returns a body that
fits the prompt budget BYTE-IDENTICALLY, so the ~770 ordinary cards in the
board's body population render exactly as before; the coverage appendix is
emitted only for a body the clamp actually cuts. Nothing is injected into a
card body: ``covers:``/``Fold-in:`` are written by the body author (the
decomposer), never fabricated here.

Note on ``Fold-in``
-------------------
The wiki doctrine (page system/kanban-decomposer-completeness-and-stamp-
propagation, rule C-5) writes the fold-in as ``Fold-in: <absorbing child> —
<why>`` while the accepted implementation plan (t_6fa2f4da §2 W-7) parses
``Fold-in: <tag> — <why>``. Both are satisfied by one canonical shape: the line
lives in the ABSORBING child's body and names the STREAM it absorbs. The
absorbing child is therefore the card carrying the line, and the tag keeps the
stream -> child mapping machine-readable. ``foldin_streams`` additionally
tolerates an explicit child reference in parentheses
(``Fold-in: S4 (into t_abcd1234) — why``) without needing it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

#: The short-form frontmatter key (``streams: S1, S2``). Do NOT rename.
STREAMS_FIELD = "streams"

#: The exact coverage keys. The prompt contract, the parser and (W-7) the
#: receipt all key off these literals — never rename or add aliases.
COVERS_KEY = "covers"
FOLDIN_KEY = "Fold-in"

#: The DECLARED-DELIVERABLE keys (W-10, card t_82eafe14). Same rule as the
#: coverage keys: exact literals, one source of truth, never aliased.
#:
#: ``deliverable: true``      — this child IS the root's deliverable;
#: ``delivers: <root-id>``    — this child delivers the named root.
#:
#: ``hermes_cli/kanban_gate.py`` imports BOTH the literals and
#: :func:`declared_deliverable` from here, so the prompt contract, the reply
#: parser and the selector cannot drift apart.
DELIVERABLE_KEY = "deliverable"
DELIVERS_KEY = "delivers"

#: Truthy spellings accepted for ``deliverable:``. CLOSED on the set: the
#: declaration is the strongest signal in the selector, so a typo must fail
#: through to the heuristic rather than silently redirect the user gate.
DELIVERABLE_TRUE_VALUES = frozenset({"true", "yes", "1"})


#: Longest description carried into the prompt appendix. The tag is what must be
#: covered; the description only has to route the reader. Bounded so a 6-stream
#: body cannot double the prompt size.
DESCRIPTION_LIMIT = 240

#: The appendix banner, so a reader can tell contract text from body text.
APPENDIX_HEADING = "DECLARED STREAM COVERAGE (extracted from the FULL root body)"

#: Emitted (loudly) when a clamped body declares no streams at all — the honest
#: signal, never an invented gap (plan §7, F3 mitigation).
NO_STREAMS_DECLARED = "NO-STREAMS-DECLARED"

# --- parsing -----------------------------------------------------------------

#: ``streams: S1, S2`` / ``- streams : S1 S2`` / ``**streams**: S1``.
_STREAMS_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*>][ \t]*)*(?:\*\*|__)?[ \t]*" + STREAMS_FIELD
    + r"[ \t]*(?:\*\*|__)?[ \t]*:[ \t]*(?P<value>.*?)[ \t]*[*`]*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

#: A markdown heading line — the STREAMS section ends at the next one.
_HEADING_RE = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]+\S")

#: A heading naming the section: ``## STREAMS``, ``## SCOPE STREAMS``,
#: ``## STREAMS TO PLAN (...)``, ``**STREAMS**``. ``\bSTREAMS?\b`` cannot match
#: inside "STREAMING" (word boundary), so a prose word is not a section.
#:
#: The line must LOOK like a heading — a markdown ``#``-heading or a bold-only
#: line — because the measured root bodies discuss streams in prose ("every
#: named stream/scope section ...", "stream→child mapping") and a bare
#: ``\bSTREAMS?\b`` anywhere on a line would open the section at the wrong
#: place and silently declare zero streams for a body that declares six.
_SECTION_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?P<lead>#{1,6}[ \t]*|\*\*|__)[ \t]*"
    r"[^\n]*?\bSTREAMS?\b[^\n]*$"
)

#: ``S1 STRUCTURAL AUDIT — ...`` / ``- **S3** SIGNALS-DB AUDIT: ...`` /
#: ``R2) EVIDENCE ...``. The tag must open the line (after bullet/bold markers)
#: and be followed by a description, so a bare mention never counts.
_TAG_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*+][ \t]*)?(?:\*\*|__)?[ \t]*(?P<tag>[A-Za-z]{1,3}\d{1,3})"
    r"(?:\*\*|__)?[ \t]*(?:[.:)\-–—][ \t]*)?(?P<desc>\S.*?)[ \t]*$",
    re.MULTILINE,
)

#: A stream tag token, canonical form: 1-3 letters + 1-3 digits.
_TAG_RE = re.compile(r"^[A-Za-z]{1,3}\d{1,3}$")

#: A ``covers: S3`` line (also ``- **covers**: S3``).
_COVERS_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*>][ \t]*)*(?:\*\*|__)?[ \t]*" + COVERS_KEY
    + r"[ \t]*(?:\*\*|__)?[ \t]*:[ \t]*(?P<value>.*?)[ \t]*[*`]*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

#: A ``Fold-in: S4 — why`` line.
_FOLDIN_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*>][ \t]*)*(?:\*\*|__)?[ \t]*" + FOLDIN_KEY
    + r"[ \t]*(?:\*\*|__)?[ \t]*:[ \t]*(?P<value>.*?)[ \t]*[*`]*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

#: Tag tokens inside a declaration value: split on commas/semicolons/slashes
#: and whitespace; parentheses keep a tolerated ``(into t_abcd1234)`` clause.
_VALUE_SPLIT_RE = re.compile(r"[,;/|]+|\s+")


@dataclass(frozen=True)
class DeclaredStream:
    """One stream the root body declares: canonical ``tag``, its char
    ``offset`` in the FULL body, and the single-line ``description``."""

    tag: str
    offset: int
    description: str


def normalize_tag(value: object) -> Optional[str]:
    """Canonical ``TAG`` (uppercase) for ``value``, else ``None``.

    Tolerates the way authors write tags in prose: ``S1``, ``s3``, ``**R2**``,
    ``S4.``, ``R5)``, ``S6:``. The enum is CLOSED on the tag SHAPE (1-3 letters
    + 1-3 digits) — anything else is rejected rather than coerced, so a typo or
    a prose word can never be read downstream as a real stream.
    """
    if not isinstance(value, str):
        return None
    token = value.strip().strip("*_`").strip()
    token = token.rstrip(".:)-–—").strip().strip("*_`").strip()
    return token.upper() if _TAG_RE.match(token) else None


def normalize_tags(value: object) -> list[str]:
    """Canonical tags found in a declaration value, in first-seen order.

    ``"S1, S2"`` -> ``["S1", "S2"]``; ``"S3 (into t_ab12) — why"`` -> ``["S3"]``.
    """
    if not isinstance(value, str) or not value.strip():
        return []
    seen: list[str] = []
    for token in _VALUE_SPLIT_RE.split(value.strip()):
        tag = normalize_tag(token)
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def _section_span(text: str) -> Optional[tuple[int, int]]:
    """``(start, end)`` char span of the first STREAMS section, else ``None``.

    ``start`` is the offset just past the section heading line; ``end`` is the
    next markdown heading, or the end of the body.
    """
    m = _SECTION_HEADING_RE.search(text)
    if m is None:
        return None
    start = m.end()
    nxt = _HEADING_RE.search(text, start)
    return start, (nxt.start() if nxt else len(text))


def declared_streams(body: object) -> list[DeclaredStream]:
    """Every stream ``body`` declares, in document order (first declaration wins).

    Two accepted forms, both declaration-driven:

    * a section whose heading contains ``STREAM``/``STREAMS``, with tagged
      entries (``S1 <description>``);
    * a ``streams: S1, S2`` frontmatter line (no descriptions).

    A body that declares nothing returns ``[]`` — absence is the signal, and it
    is never turned into an invented gap.
    """
    text = body if isinstance(body, str) else ""
    if not text.strip():
        return []

    found: dict[str, DeclaredStream] = {}
    order: list[str] = []

    span = _section_span(text)
    if span is not None:
        start, end = span
        for m in _TAG_LINE_RE.finditer(text, start, end):
            tag = normalize_tag(m.group("tag"))
            if not tag or tag in found:
                continue
            desc = re.sub(r"\s+", " ", m.group("desc")).strip()
            found[tag] = DeclaredStream(tag=tag, offset=m.start("tag"), description=desc)
            order.append(tag)

    for m in _STREAMS_LINE_RE.finditer(text):
        for tag in normalize_tags(m.group("value")):
            if tag in found:
                continue
            found[tag] = DeclaredStream(tag=tag, offset=m.start("value"), description="")
            order.append(tag)

    return [found[t] for t in order]


def declared_stream_tags(body: object) -> list[str]:
    """Just the canonical tags ``body`` declares (``[]`` when it declares none)."""
    return [s.tag for s in declared_streams(body)]


def covers_tags(text: object) -> list[str]:
    """Stream tags declared by ``covers:`` lines in ``text`` (a child body)."""
    if not isinstance(text, str) or not text.strip():
        return []
    found: list[str] = []
    for m in _COVERS_LINE_RE.finditer(text):
        for tag in normalize_tags(m.group("value")):
            if tag not in found:
                found.append(tag)
    return found


def foldin_streams(text: object) -> list[str]:
    """Stream tags declared by ``Fold-in:`` lines in ``text`` (the absorbing child)."""
    if not isinstance(text, str) or not text.strip():
        return []
    found: list[str] = []
    for m in _FOLDIN_LINE_RE.finditer(text):
        for tag in normalize_tags(m.group("value")):
            if tag not in found:
                found.append(tag)
    return found


def accounted_tags(text: object) -> list[str]:
    """Tags ``text`` accounts for, by ``covers:`` or ``Fold-in:``."""
    found = covers_tags(text)
    for tag in foldin_streams(text):
        if tag not in found:
            found.append(tag)
    return found


#: ``deliverable: true`` / ``- **deliverable**: true``.
_DELIVERABLE_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*>][ \t]*)*(?:\*\*|__)?[ \t]*" + DELIVERABLE_KEY
    + r"[ \t]*(?:\*\*|__)?[ \t]*:[ \t]*(?P<value>.*?)[ \t]*[*`]*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

#: ``delivers: t_abcd1234``.
_DELIVERS_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*>][ \t]*)*(?:\*\*|__)?[ \t]*" + DELIVERS_KEY
    + r"[ \t]*(?:\*\*|__)?[ \t]*:[ \t]*(?P<value>.*?)[ \t]*[*`]*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def declared_deliverable(body: object) -> Optional[str]:
    """The declared-deliverable VALUE on ``body``, else ``None`` (W-10).

    Two accepted forms, both exact-literal declaration lines:

    * ``deliverable: true``      -> returns ``"true"``
    * ``delivers: <root-id>``    -> returns the root id

    Declaration-driven, exactly like ``covers:``/``Fold-in:``: a bare *prose*
    mention of the word deliverable is NOT a declaration, so the gate selector
    can never be steered by text mining. ``deliverable: false`` (or any value
    outside :data:`DELIVERABLE_TRUE_VALUES`) is treated as absent — an explicit
    "not me" must fall through to the heuristic rather than mis-read as a hit.
    ``delivers:`` wins over ``deliverable:`` when both lines are present.
    """
    if not isinstance(body, str) or not body.strip():
        return None
    m = _DELIVERS_LINE_RE.search(body)
    if m is not None:
        value = m.group("value").strip().strip("*`").strip()
        if value:
            return value
    m = _DELIVERABLE_LINE_RE.search(body)
    if m is not None:
        value = m.group("value").strip().strip("*`").strip().lower()
        if value in DELIVERABLE_TRUE_VALUES:
            return "true"
    return None


def deliverable_lines(*, deliverable: object = None, delivers: object = None) -> list[str]:
    """Canonical declaration line(s) for a child that IS the deliverable.

    ``deliverable`` is the truthy toggle (``True`` -> ``deliverable: true``);
    ``delivers`` is the root id (``delivers: t_abcd1234``). Returns ``[]`` for
    an absent/false value — a declaration is never invented.
    """
    lines: list[str] = []
    if isinstance(delivers, str) and delivers.strip():
        lines.append(f"{DELIVERS_KEY}: {delivers.strip()}")
    value = deliverable
    if isinstance(value, str):
        value = value.strip().lower()
    if value is True or (isinstance(value, str) and value in DELIVERABLE_TRUE_VALUES):
        lines.append(f"{DELIVERABLE_KEY}: true")
    return lines


def ensure_deliverable_line(body: object, *, deliverable: object = None,
                            delivers: object = None) -> str:
    """``body`` with the deliverable declaration folded in at most once.

    Idempotent and non-fabricating, mirroring :func:`ensure_declaration_lines`:
    a body that already declares one keeps its bytes, and a body handed
    nothing is returned byte-identically — so the ordinary (undeclared) fan-out
    is untouched.
    """
    text = body if isinstance(body, str) else ""
    if declared_deliverable(text) is not None:
        return text
    lines = deliverable_lines(deliverable=deliverable, delivers=delivers)
    if not lines:
        return text
    block = "\n".join(lines)
    return f"{text.rstrip()}\n\n{block}" if text.strip() else block



def missing_coverage(root_body: object, child_bodies: Iterable[object]) -> list[str]:
    """Declared streams no child accounts for — the declaration oracle.

    This is the primitive W-7's receipt reports on. It is deliberately NOT a
    text/keyword diff: only an explicit ``covers:``/``Fold-in:`` declaration
    counts as coverage, which is what makes it score 0/4 false-passes on the
    motivating incident where a keyword diff scored 4/4.

    ``child_bodies`` with no declaration of any kind return the full declared
    set — the honest answer for the pre-contract population, and the reason the
    receipt must never be built from the prompt slice (a sliced root declares
    fewer streams and can pass vacuously).
    """
    declared = declared_stream_tags(root_body)
    covered: set[str] = set()
    for child in child_bodies:
        covered.update(accounted_tags(child))
    return [tag for tag in declared if tag not in covered]


# --- declaration rendering ---------------------------------------------------


def _foldin_reason_map(value: object) -> list[tuple[str, str]]:
    """``value`` as ``[(tag, reason)]``; tolerant of str / list / dict shapes.

    A model may answer with ``"S4"``, ``["S4", "S5"]``, ``"S4 — absorbed by the
    census card"``, ``"S4: absorbed"``, or ``{"S4": "why"}``. An unknown/empty
    value yields ``[]`` so nothing is ever fabricated from a malformed
    declaration.
    """
    out: list[tuple[str, str]] = []

    def _push(raw_tag: object, reason: str) -> None:
        tag = normalize_tag(raw_tag)
        if tag and all(tag != t for t, _ in out):
            out.append((tag, reason))

    if isinstance(value, dict):
        for k, v in value.items():
            _push(k, str(v).strip() if v else "")
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            for tag, reason in _foldin_reason_map(item):
                _push(tag, reason)
    elif isinstance(value, str) and value.strip():
        # Split a comma/semicolon list into independent entries; a single
        # entry keeps any "— why" tail as its reason.
        parts = [p for p in re.split(r"[,;]+", value.strip()) if p.strip()]
        for part in parts or [value]:
            text = part.strip()
            m = re.match(r"^(?P<tag>[A-Za-z]{1,3}\d{1,3})\b(?P<rest>.*)$", text)
            if m:
                reason = re.sub(r"^[\s:—\-–]+", "", m.group("rest")).strip()
                _push(m.group("tag"), reason)
            else:
                # Not tag-led: it may still START with a tag after a symbol.
                _push(text.split()[0] if text.split() else "",
                      " ".join(text.split()[1:]))
    return out


def coverage_lines(*, covers: object = None, fold_in: object = None) -> list[str]:
    """Canonical declaration lines for a child body, one per declared stream.

    ``covers`` takes tags (str/list); ``fold_in`` takes tags with an optional
    reason (str/list/dict). Returns ``[]`` for an absent/empty/unparseable
    value — a declaration is never invented.
    """
    lines: list[str] = []
    for tag in normalize_tags(covers) if isinstance(covers, str) else [
        t for item in (covers or []) for t in normalize_tags(item)
    ]:
        lines.append(f"{COVERS_KEY}: {tag}")
    for tag, reason in _foldin_reason_map(fold_in):
        why = reason or "absorbed into this card's deliverable"
        lines.append(f"{FOLDIN_KEY}: {tag} — {why}")
    return lines


def ensure_declaration_lines(body: object, *, covers: object = None,
                             fold_in: object = None) -> str:
    """``body`` with the caller's declarations rendered at most once per stream.

    Idempotent and non-fabricating: a tag the body already declares under the
    same key is skipped, and a body handed nothing is returned byte-identically.
    This is what lets the decomposer's structured reply keys (``covers`` /
    ``fold_in``) OR a hand-written body line survive into the persisted child
    body — the parser accepts either, so the mapping is machine-readable
    whichever shape the model chose (W-8 AC2).
    """
    text = body if isinstance(body, str) else ""
    have_covers = set(covers_tags(text))
    have_fold = set(foldin_streams(text))
    add: list[str] = []
    for line in coverage_lines(covers=covers, fold_in=fold_in):
        if line.lower().startswith(COVERS_KEY.lower()):
            tag = normalize_tag(line.split(":", 1)[1])
            if not tag or tag in have_covers:
                continue
            have_covers.add(tag)
        elif line.lower().startswith(FOLDIN_KEY.lower()):
            tag = normalize_tag(line.split(":", 1)[1].split("—")[0])
            if not tag or tag in have_fold:
                continue
            have_fold.add(tag)
        add.append(line)
    if not add:
        return text
    block = "\n".join(add)
    return f"{text.rstrip()}\n\n{block}" if text.strip() else block


# --- prompt projection -------------------------------------------------------


def _clip(text: str, limit: int = DESCRIPTION_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def coverage_appendix(full_body: str, *, cut: int) -> str:
    """The declared-coverage block appended below a clamped body.

    States the truncation fact (cut offset of the full length) and lists every
    declared stream with its ``@char`` offset and whether it lies inside or
    beyond the cut — so a stream past the cut is now SEEN, and the reader can
    separate a compliance failure from a context-boundary failure (plan §7.2).
    """
    streams = declared_streams(full_body)
    out = ["", f"--- {APPENDIX_HEADING} ---"]
    out.append(
        f"The body above is CUT at char {cut} of {len(full_body)}; the text past "
        f"that point was NOT shown to you."
    )
    if not streams:
        out.append(
            f"{NO_STREAMS_DECLARED}: this body declares no `## STREAMS` section and "
            f"no `{STREAMS_FIELD}:` line, so no coverage can be declared for it."
        )
        out.append(
            "Decompose from the visible text only, and do NOT invent stream tags."
        )
    else:
        out.append(
            f"The root DECLARES {len(streams)} stream(s). Every one of them must be "
            f"covered or explicitly folded in:"
        )
        out.append("")
        for s in streams:
            place = (
                "inside the visible text"
                if s.offset < cut
                else "BEYOND the cut (NOT visible above)"
            )
            desc = _clip(s.description)
            out.append(f"  [{s.tag}] @char {s.offset} — {place} — {desc}" if desc
                       else f"  [{s.tag}] @char {s.offset} — {place}")
        out.append("")
        out.append(
            f"For every stream a card you emit OWNS, write a line `{COVERS_KEY}: <tag>` "
            f"in that card's body."
        )
        out.append(
            f"For a stream you absorb into another card's deliverable, write "
            f"`{FOLDIN_KEY}: <tag> — <why>` in THAT absorbing card's body."
        )
        out.append(
            "A declared stream with neither declaration is a COVERAGE GAP and the "
            "fan-out is incomplete."
        )
    out.append(f"--- END DECLARED STREAM COVERAGE ---")
    return "\n".join(out)


def prompt_body(body: object, *, limit: int) -> str:
    """The body field of the decomposer/specifier user prompt.

    Byte-identical to ``body`` when it fits ``limit`` (the ordinary case: ~770
    board bodies). A body the clamp would cut is still clamped exactly as
    before, then carries :func:`coverage_appendix` — the declared streams read
    from the FULL body, so completeness stops depending on where the cut lands.
    """
    text = body if isinstance(body, str) else ""
    if len(text) <= limit:
        return text
    cut = limit - 1
    clamped = text[:cut] + "…"
    return f"{clamped}\n\n{coverage_appendix(text, cut=cut)}"


def coverage_prompt_block() -> str:
    """The contract block the card templates render from (single source of truth).

    Rendered from the constants above so the specifier prompt, the decomposer
    prompt, the parser and (W-7) the receipt cannot drift apart.
    """
    return f"""
DECLARED STREAM COVERAGE (mandatory whenever the root declares streams)
  A root body may DECLARE its streams in a section whose heading contains the
  word STREAM/STREAMS, as tagged entries:

      ## SCOPE STREAMS (triager to decompose)
      S1 <description>
      S2 <description>

  (a `{STREAMS_FIELD}: S1, S2` frontmatter line is the accepted short form.)

  The user message lists EVERY declared stream — tag, `@char` offset and whether
  it lies inside or BEYOND the (possibly clamped) body — so a stream the raw
  body text would have cut off is still in front of you. When that list is
  present, the coverage contract is BINDING:

  - Each declared stream must be accounted for by exactly ONE emitted card:
      * write `{COVERS_KEY}: <tag>` in the body of the card that OWNS the
        stream's deliverable (repeat the line, once per stream, when a card
        owns several); or
      * write `{FOLDIN_KEY}: <tag> — <why>` in the body of the card that
        ABSORBS the stream into its own deliverable (rule C-5: an intentional
        fold-in is a first-class disposition, and it is also where "descoped by
        the user at triage" is recorded).
  - The same declaration may be given STRUCTURALLY on the task object instead of
    as a body line — `"covers": ["S3"]` and `"fold_in": {{"S4": "<why>"}}` — and
    it is folded into that card's body for you. Emit ONE of the two shapes per
    stream, never both.
  - The keys are EXACT: `{COVERS_KEY}:` and `{FOLDIN_KEY}:`, one declaration per
    line. Do not rename them, do not fold them into prose.
  - NEVER fabricate a `{COVERS_KEY}:` line for a stream a card does not own, and
    never drop a declared stream silently: a stream with neither declaration is
    a COVERAGE GAP, the fan-out is incomplete, and a reviewer will reject it.
  - A body that declares NO streams gets no coverage lines at all — absence is
    the signal; do not invent stream tags for it.
  - With `fanout=false` (a single card), that one card must carry the
    declarations for every declared stream.
"""


def deliverable_prompt_block() -> str:
    """The DECLARED-DELIVERABLE contract block (W-10, card t_82eafe14).

    Rendered from :data:`DELIVERABLE_KEY` / :data:`DELIVERS_KEY` so the
    decomposer prompt, the reply parser (``kanban_decompose``) and the gate
    selector (``kanban_gate.select_deliverable``) share one source of truth.

    Why this exists: the user gate reaches exactly ONE child per root (1-of-N by
    construction), and the phase-order heuristic that chose it is structurally
    biased — the terminal chain phase is ``testing``, so the gate landed on a
    *verification* card on both incident roots (measured: 21/21 gated roots
    reproduced by the heuristic, but fallback-decided and silent). A declaration
    is the only signal that names the deliverable on purpose.
    """
    return f"""
DECLARED DELIVERABLE (mandatory whenever the quest is user-gated)
  A root that requires a user verdict routes that gate to exactly ONE child —
  the DELIVERABLE. Do not leave that choice to a title/phase heuristic: mark it
  explicitly, on the task object AND/OR as an exact body line:

      "{DELIVERABLE_KEY}": true          ->  {DELIVERABLE_KEY}: true
      "{DELIVERS_KEY}": "<root task id>"  ->  {DELIVERS_KEY}: <root task id>

  - Write it in the body of the child whose work IS the root's deliverable —
    the synthesis / consolidated report / final verdict card, NOT a verification,
    research or wiki-landing card.
  - EXACTLY ONE child may carry it for a given root. On every other child, omit
    the `{DELIVERABLE_KEY}` key entirely (do not send `false`). If two children
    somehow declare it, the FIRST emitted one wins and the fan-out is ambiguous —
    that is a defect, so declare once.
  - The keys are EXACT: `{DELIVERABLE_KEY}:` and `{DELIVERS_KEY}:`. Never rename
    them, never fold them into prose, and never mark more than one child.
  - `{DELIVERABLE_KEY}: false` means "not me" and is honoured as an explicit
    non-declaration — but do not write it; simply omit the line.
  - Without a declaration the gate falls back to a title/phase heuristic and
    that fallback is recorded as such (a reviewer sees a fallback-decided
    recipient as a defect). Prefer the declaration.
"""


__all__ = [
    "APPENDIX_HEADING",
    "COVERS_KEY",
    "DELIVERABLE_KEY",
    "DELIVERABLE_TRUE_VALUES",
    "DELIVERS_KEY",
    "DeclaredStream",
    "DESCRIPTION_LIMIT",
    "FOLDIN_KEY",
    "NO_STREAMS_DECLARED",
    "STREAMS_FIELD",
    "accounted_tags",
    "covers_tags",
    "coverage_appendix",
    "coverage_lines",
    "coverage_prompt_block",
    "declared_deliverable",
    "declared_stream_tags",
    "declared_streams",
    "deliverable_lines",
    "deliverable_prompt_block",
    "ensure_declaration_lines",
    "ensure_deliverable_line",
    "foldin_streams",
    "missing_coverage",
    "normalize_tag",
    "normalize_tags",
    "prompt_body",
]
