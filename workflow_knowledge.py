"""Long-term knowledge shared across tasks: a reviewed, file-based store.

The journal (``Task.events``) and active memory belong to one task. A
``KnowledgeStore`` holds what should outlive tasks: facts and lessons, kept as
*pages* of short *entries* in a host-owned folder. There are no embeddings and
no vector index: models see a generated index of page titles and read pages on
demand, or search them by keyword.

Layout of the store folder::

    pages/s-<scope>/p-<slug>.json   one page; the source of truth
    proposals/<id>.json             requested changes and their review status
    log.jsonl                       append-only record of applied changes
    usage.json                      how often each page was read

Writing is deliberately indirect. Nothing edits a page in place:

1. a change is *proposed* as a list of small operations (create a page, add,
   update or retire an entry), each proposal carrying its source and evidence;
2. the operations are validated and dry-run against the current pages;
3. on approval they are applied by ordinary code, the page version increases,
   and the before/after documents go to the log so the change can be reverted.

Small operations merged by code, rather than a model rewriting a page, keep
knowledge from being silently shortened or replaced. Review and provenance are
the defence against content that was injected into a task and tries to persist.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock

from workflow_memory_store import atomic_write
from workflow_types import JsonDict
from workflow_utils import new_id

SCOPE_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9_.-]{0,62}[a-z0-9])?")
SLUG_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")

OPERATIONS = ("create_page", "set_summary", "add_entry", "update_entry", "retire_entry")
SOURCE_KINDS = ("internal", "external", "host")
PROPOSAL_STATUSES = ("pending", "applied", "rejected")

MAX_OPERATIONS = 20
MAX_TITLE_CHARS = 120
MAX_SUMMARY_CHARS = 300
MAX_ENTRY_CHARS = 1000
MAX_RATIONALE_CHARS = 1000
MAX_ENTRIES_PER_PAGE = 200

# Lint thresholds: a page past these is still valid, but worth splitting.
SOFT_ENTRIES_PER_PAGE = 50
SOFT_PAGE_CHARS = 8000

LOCK_TIMEOUT_SECONDS = 10
LOCK_STALE_SECONDS = 60

# Words, or one Japanese kana / CJK ideograph / Korean syllable at a time.
_WORD = re.compile(r"[a-z0-9]+|[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def page_key(scope: str, slug: str) -> str:
    return f"{scope}/{slug}"


def split_key(key: str) -> tuple[str, str]:
    """Split and validate a ``"<scope>/<slug>"`` page key."""
    if not isinstance(key, str) or key.count("/") != 1:
        raise ValueError("A page is named '<scope>/<slug>'.")
    scope, slug = key.split("/")
    check_scope(scope)
    if not SLUG_PATTERN.fullmatch(slug):
        raise ValueError(f"Invalid page slug {slug!r}: use lower-case letters, digits and hyphens.")
    return scope, slug


def check_scope(scope: str) -> str:
    if not isinstance(scope, str) or not SCOPE_PATTERN.fullmatch(scope):
        raise ValueError(
            f"Invalid scope {scope!r}: use lower-case letters, digits, '.', '_' and '-'."
        )
    return scope


def tokenize(text: str) -> list[str]:
    """Lower-case words, plus single characters and adjacent pairs for CJK text."""
    tokens: list[str] = []
    previous = None
    for match in _WORD.finditer(text.lower()):
        token = match.group()
        tokens.append(token)
        is_cjk = not token.isascii()
        if is_cjk and previous is not None and previous.end() == match.start():
            tokens.append(previous.group() + token)
        previous = match if is_cjk else None
    return tokens


def _normalized(text: str) -> str:
    return " ".join(tokenize(text))


def _check_text(value, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text.")
    if len(value) > limit:
        raise ValueError(f"{name} is longer than {limit} characters.")
    if _CONTROL.search(value):
        raise ValueError(f"{name} contains control characters.")
    return value.strip()


def _active(page: JsonDict) -> list[JsonDict]:
    return [entry for entry in page["entries"] if entry["status"] == "active"]


def render_page(page: JsonDict, *, include_retired: bool = False) -> str:
    """Readable text for a model or a person; entry IDs are shown for updates."""
    key = page_key(page["scope"], page["slug"])
    lines = [f"# {page['title']} ({key}, version {page['version']})", page["summary"], ""]
    for entry in page["entries"]:
        if entry["status"] == "active":
            lines.append(f"- [{entry['id']}] {entry['text']}")
        elif include_retired:
            reason = entry.get("retired_reason", "")
            lines.append(f"- [{entry['id']}] (retired: {reason}) {entry['text']}")
    return "\n".join(lines)


class KnowledgeStore:
    """File-backed knowledge base. Safe for several processes sharing the folder."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "pages").mkdir(parents=True, exist_ok=True)
        (self.root / "proposals").mkdir(exist_ok=True)
        self._thread_lock = RLock()
        self._lock_depth = 0

    def __repr__(self) -> str:
        return f"KnowledgeStore({str(self.root)!r})"

    # ------------------------------------------------------------------
    # Locking and files
    # ------------------------------------------------------------------

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Exclusive access for writers, across threads and processes."""
        with self._thread_lock:
            if self._lock_depth:
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return
            lock = self.root / ".lock"
            deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    break
                except FileExistsError:
                    try:
                        if time.time() - lock.stat().st_mtime > LOCK_STALE_SECONDS:
                            lock.unlink(missing_ok=True)
                            continue
                    except FileNotFoundError:
                        continue
                    if time.monotonic() > deadline:
                        raise TimeoutError("The knowledge store is locked by another writer.")
                    time.sleep(0.05)
            self._lock_depth = 1
            try:
                yield
            finally:
                self._lock_depth = 0
                os.close(descriptor)
                lock.unlink(missing_ok=True)

    def _page_path(self, key: str) -> Path:
        scope, slug = split_key(key)
        return self.root / "pages" / f"s-{scope}" / f"p-{slug}.json"

    def _proposal_path(self, proposal_id: str) -> Path:
        if not isinstance(proposal_id, str) or not re.fullmatch(
            r"proposal_[0-9a-f]{12}", proposal_id
        ):
            raise KeyError(f"Unknown proposal: {proposal_id!r}")
        return self.root / "proposals" / f"{proposal_id}.json"

    @staticmethod
    def _read_json(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, document) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8"))

    def _load(self, key: str) -> JsonDict | None:
        path = self._page_path(key)
        return self._read_json(path) if path.exists() else None

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def pages(self, scopes: Iterable[str] | None = None) -> list[JsonDict]:
        """All page documents, optionally limited to some scopes, sorted by key."""
        wanted = None if scopes is None else {check_scope(scope) for scope in scopes}
        found = []
        for path in sorted((self.root / "pages").glob("s-*/p-*.json")):
            if wanted is None or path.parent.name[2:] in wanted:
                found.append(self._read_json(path))
        return found

    def index(self, scopes: Iterable[str] | None = None) -> list[JsonDict]:
        """One line per page: what a model sees before deciding what to read."""
        return [
            {
                "page": page_key(page["scope"], page["slug"]),
                "title": page["title"],
                "summary": page["summary"],
                "entries": len(_active(page)),
                "updated_at": page["updated_at"],
            }
            for page in self.pages(scopes)
        ]

    def read_page(self, key: str) -> JsonDict:
        page = self._load(key)
        if page is None:
            raise KeyError(f"Unknown knowledge page: {key!r}")
        return page

    def search(
        self, query: str, scopes: Iterable[str] | None = None, limit: int = 10
    ) -> list[JsonDict]:
        """Keyword search (BM25) over active entries. Deterministic; no embeddings."""
        terms = set(tokenize(query))
        if not terms:
            raise ValueError("The search query has no searchable words.")
        documents = []
        for page in self.pages(scopes):
            title_tokens = tokenize(page["title"])
            for entry in _active(page):
                tokens = tokenize(entry["text"]) + title_tokens
                documents.append((page, entry, Counter(tokens), len(tokens)))
        if not documents:
            return []
        average = sum(length for *_, length in documents) / len(documents)
        frequency = Counter(
            term for _, _, counts, _ in documents for term in terms if term in counts
        )
        results = []
        for page, entry, counts, length in documents:
            score = 0.0
            for term in terms:
                occurrences = counts.get(term, 0)
                if not occurrences:
                    continue
                rarity = math.log(
                    1 + (len(documents) - frequency[term] + 0.5) / (frequency[term] + 0.5)
                )
                saturation = (
                    occurrences * 2.5 / (occurrences + 1.5 * (0.25 + 0.75 * length / average))
                )
                score += rarity * saturation
            if score > 0:
                results.append(
                    {
                        "page": page_key(page["scope"], page["slug"]),
                        "title": page["title"],
                        "entry": entry["id"],
                        "text": entry["text"],
                        "score": round(score, 4),
                    }
                )
        results.sort(key=lambda hit: (-hit["score"], hit["page"], hit["entry"]))
        return results[:limit]

    # ------------------------------------------------------------------
    # Planning changes
    # ------------------------------------------------------------------

    @staticmethod
    def check_operations(operations) -> list[JsonDict]:
        """Structural validation of a list of operations (no store access)."""
        if not isinstance(operations, list) or not operations:
            raise ValueError("operations must be a nonempty list.")
        if len(operations) > MAX_OPERATIONS:
            raise ValueError(f"A change may contain at most {MAX_OPERATIONS} operations.")
        fields = {
            "create_page": {"title", "summary"},
            "set_summary": {"summary"},
            "add_entry": {"text"},
            "update_entry": {"entry", "text"},
            "retire_entry": {"entry", "reason"},
        }
        optional = {"set_summary": {"title"}}
        for number, operation in enumerate(operations, 1):
            if not isinstance(operation, dict) or operation.get("op") not in OPERATIONS:
                raise ValueError(f"Operation {number}: 'op' must be one of {OPERATIONS}.")
            name = operation["op"]
            required = {"op", "page"} | fields[name]
            allowed = required | optional.get(name, set())
            if not required <= operation.keys() or not operation.keys() <= allowed:
                raise ValueError(
                    f"Operation {number} ({name}): expected fields {sorted(required)}."
                )
            split_key(operation["page"])
        return copy.deepcopy(operations)

    def _plan(self, operations, source: JsonDict, reviewer: str, change_id: str):
        """Dry-run the operations. Returns ``{key: (before, after)}`` for changed pages."""
        operations = self.check_operations(operations)
        now = _now()
        before: dict[str, JsonDict | None] = {}
        working: dict[str, JsonDict | None] = {}

        def page_for(key: str, *, must_exist: bool = True) -> JsonDict | None:
            if key not in working:
                before[key] = self._load(key)
                working[key] = copy.deepcopy(before[key])
            if must_exist and working[key] is None:
                raise KeyError(f"page {key!r} does not exist; create it first")
            return working[key]

        def entry_for(page: JsonDict, entry_id: str) -> JsonDict:
            for entry in _active(page):
                if entry["id"] == entry_id:
                    return entry
            raise KeyError(f"no active entry {entry_id!r} on this page")

        def stamp(entry: JsonDict) -> None:
            entry.update(updated_at=now, source=copy.deepcopy(source), reviewer=reviewer)
            entry["change_id"] = change_id

        for number, operation in enumerate(operations, 1):
            name, key = operation["op"], operation["page"]
            try:
                if name == "create_page":
                    if page_for(key, must_exist=False) is not None:
                        raise ValueError(f"page {key!r} already exists")
                    scope, slug = split_key(key)
                    working[key] = {
                        "scope": scope,
                        "slug": slug,
                        "title": _check_text(operation["title"], "title", MAX_TITLE_CHARS),
                        "summary": _check_text(operation["summary"], "summary", MAX_SUMMARY_CHARS),
                        "version": 0,
                        "created_at": now,
                        "updated_at": now,
                        "entries": [],
                    }
                    continue
                page = page_for(key)
                if name == "set_summary":
                    page["summary"] = _check_text(
                        operation["summary"], "summary", MAX_SUMMARY_CHARS
                    )
                    if "title" in operation:
                        page["title"] = _check_text(operation["title"], "title", MAX_TITLE_CHARS)
                elif name == "add_entry":
                    text = _check_text(operation["text"], "text", MAX_ENTRY_CHARS)
                    if len(_active(page)) >= MAX_ENTRIES_PER_PAGE:
                        raise ValueError(
                            f"page {key!r} is full; retire entries or start a new page"
                        )
                    duplicate = next(
                        (e for e in _active(page) if _normalized(e["text"]) == _normalized(text)),
                        None,
                    )
                    if duplicate is not None:
                        raise ValueError(f"same text as existing entry {duplicate['id']!r}")
                    entry = {"id": new_id("k"), "text": text, "status": "active", "created_at": now}
                    stamp(entry)
                    page["entries"].append(entry)
                elif name == "update_entry":
                    entry = entry_for(page, operation["entry"])
                    entry["text"] = _check_text(operation["text"], "text", MAX_ENTRY_CHARS)
                    stamp(entry)
                else:  # retire_entry
                    entry = entry_for(page, operation["entry"])
                    entry["status"] = "retired"
                    entry["retired_reason"] = _check_text(
                        operation["reason"], "reason", MAX_SUMMARY_CHARS
                    )
                    stamp(entry)
            except (KeyError, ValueError) as exc:
                message = exc.args[0] if exc.args else str(exc)
                raise ValueError(f"Operation {number} ({name}): {message}") from exc

        changed = {}
        for key, after in working.items():
            if after != before[key]:
                after["version"] = (before[key]["version"] if before[key] else 0) + 1
                after["updated_at"] = now
                changed[key] = (before[key], after)
        return changed

    @staticmethod
    def _check_source(source) -> JsonDict:
        if not isinstance(source, dict) or source.get("kind") not in SOURCE_KINDS:
            raise ValueError(f"source.kind must be one of {SOURCE_KINDS}.")
        json.dumps(source, allow_nan=False)
        return copy.deepcopy(source)

    # ------------------------------------------------------------------
    # Proposals and review
    # ------------------------------------------------------------------

    def propose(
        self, operations, *, rationale: str, source: JsonDict, auto_apply: bool = False
    ) -> JsonDict:
        """Queue a change for review. Invalid changes are refused here, not later.

        ``auto_apply`` applies the change immediately with reviewer ``"auto"``;
        callers decide when that is acceptable.
        """
        source = self._check_source(source)
        rationale = _check_text(rationale, "rationale", MAX_RATIONALE_CHARS)
        with self._locked():
            proposal = {
                "id": new_id("proposal"),
                "status": "pending",
                "created_at": _now(),
                "operations": self.check_operations(operations),
                "rationale": rationale,
                "source": source,
                "reviewer": None,
                "comment": "",
                "resolved_at": None,
                "change_id": None,
            }
            self._plan(proposal["operations"], source, "dry-run", "dry-run")
            self._write_json(self._proposal_path(proposal["id"]), proposal)
            if auto_apply:
                return self.resolve_proposal(proposal["id"], True, reviewer="auto")
            return proposal

    def proposals(self, status: str | None = "pending") -> list[JsonDict]:
        """Proposals with the given status (``None`` for all), oldest first."""
        if status is not None and status not in PROPOSAL_STATUSES:
            raise ValueError(f"status must be one of {PROPOSAL_STATUSES} or None.")
        found = [
            self._read_json(path) for path in (self.root / "proposals").glob("proposal_*.json")
        ]
        found.sort(key=lambda proposal: (proposal["created_at"], proposal["id"]))
        return [p for p in found if status is None or p["status"] == status]

    def resolve_proposal(
        self, proposal_id: str, approved: bool, *, reviewer: str, comment: str = ""
    ) -> JsonDict:
        """Record a reviewer's decision; approval applies the change.

        If the pages changed so that the proposal no longer applies, this raises
        ``ValueError`` and the proposal stays pending, to be rejected or redone.
        """
        if type(approved) is not bool:
            raise TypeError("approved must be a Boolean.")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("A reviewer identity is required.")
        if not isinstance(comment, str):
            raise TypeError("comment must be a string.")
        with self._locked():
            path = self._proposal_path(proposal_id)
            if not path.exists():
                raise KeyError(f"Unknown proposal: {proposal_id!r}")
            proposal = self._read_json(path)
            if proposal["status"] != "pending":
                raise ValueError(f"Proposal {proposal_id!r} is already {proposal['status']}.")
            if approved:
                change = self._apply(
                    proposal["operations"], proposal["source"], reviewer, proposal_id=proposal_id
                )
                proposal["change_id"] = change["id"]
            proposal.update(
                status="applied" if approved else "rejected",
                reviewer=reviewer,
                comment=comment,
                resolved_at=_now(),
            )
            self._write_json(path, proposal)
            return proposal

    # ------------------------------------------------------------------
    # Applying, logging and reverting
    # ------------------------------------------------------------------

    def _write_pages(self, pages: dict[str, JsonDict | None]) -> None:
        for key, document in pages.items():
            path = self._page_path(key)
            if document is None:
                path.unlink(missing_ok=True)
            else:
                self._write_json(path, document)

    def _append_log(self, record: JsonDict) -> None:
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        with (self.root / "log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def _apply(self, operations, source, reviewer: str, *, proposal_id: str | None) -> JsonDict:
        change_id = new_id("change")
        changed = self._plan(operations, source, reviewer, change_id)
        record = {
            "id": change_id,
            "timestamp": _now(),
            "kind": "change",
            "proposal_id": proposal_id,
            "reviewer": reviewer,
            "source": source,
            "pages": {key: {"before": b, "after": a} for key, (b, a) in changed.items()},
        }
        # Log first: a crash can leave a logged change unapplied, never an unlogged one.
        self._append_log(record)
        self._write_pages({key: after for key, (_, after) in changed.items()})
        return record

    def apply(self, operations, *, author: str, rationale: str = "Host edit") -> JsonDict:
        """Trusted host edit: applied immediately, recorded as an applied proposal."""
        if not isinstance(author, str) or not author.strip():
            raise ValueError("An author identity is required.")
        with self._locked():
            proposal = self.propose(
                operations, rationale=rationale, source={"kind": "host", "author": author}
            )
            return self.resolve_proposal(proposal["id"], True, reviewer=author)

    def changes(self, *, include_pages: bool = False) -> list[JsonDict]:
        """Applied changes and reverts, oldest first."""
        log = self.root / "log.jsonl"
        if not log.exists():
            return []
        records = [
            json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line
        ]
        if not include_pages:
            for record in records:
                record["pages"] = sorted(record["pages"])
        return records

    def revert(self, change_id: str, *, reviewer: str) -> JsonDict:
        """Undo one change. Refused if a touched page was changed again afterwards."""
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("A reviewer identity is required.")
        with self._locked():
            record = next(
                (r for r in self.changes(include_pages=True) if r["id"] == change_id), None
            )
            if record is None:
                raise KeyError(f"Unknown change: {change_id!r}")
            restored: dict[str, JsonDict | None] = {}
            pages = {}
            for key, versions in record["pages"].items():
                current = self._load(key)
                if current != versions["after"]:
                    raise ValueError(
                        f"Page {key!r} changed after {change_id!r}; revert the later change first."
                    )
                document = copy.deepcopy(versions["before"])
                if document is not None:
                    document["version"] = current["version"] + 1
                    document["updated_at"] = _now()
                restored[key] = document
                pages[key] = {"before": current, "after": document}
            revert_record = {
                "id": new_id("change"),
                "timestamp": _now(),
                "kind": "revert",
                "reverts": change_id,
                "proposal_id": None,
                "reviewer": reviewer,
                "source": {"kind": "host", "author": reviewer},
                "pages": pages,
            }
            self._append_log(revert_record)
            self._write_pages(restored)
            return revert_record

    # ------------------------------------------------------------------
    # Usage and maintenance
    # ------------------------------------------------------------------

    def usage(self) -> JsonDict:
        path = self.root / "usage.json"
        return self._read_json(path) if path.exists() else {}

    def record_use(self, keys: Iterable[str]) -> None:
        """Note that these pages were read; lint uses this to find unused pages."""
        keys = sorted(set(keys))
        if not keys:
            return
        with self._locked():
            usage = self.usage()
            for key in keys:
                counter = usage.setdefault(key, {"reads": 0, "last_read": None})
                counter["reads"] += 1
                counter["last_read"] = _now()
            self._write_json(self.root / "usage.json", usage)

    def lint(self, *, unused_days: int = 90, stale_proposal_days: int = 14) -> list[JsonDict]:
        """Deterministic health checks. Returns findings; changes nothing."""
        now = datetime.now(timezone.utc)

        def older_than(timestamp: str | None, days: int) -> bool:
            return timestamp is None or now - datetime.fromisoformat(timestamp) > timedelta(
                days=days
            )

        findings: list[JsonDict] = []
        usage = self.usage()
        seen: dict[tuple[str, str], tuple[str, str]] = {}
        for page in self.pages():
            key = page_key(page["scope"], page["slug"])
            active = _active(page)
            if not active:
                findings.append({"kind": "empty_page", "page": key, "detail": "no active entries"})
            size = len(render_page(page))
            if len(active) > SOFT_ENTRIES_PER_PAGE or size > SOFT_PAGE_CHARS:
                findings.append(
                    {
                        "kind": "oversized_page",
                        "page": key,
                        "detail": f"{len(active)} entries, {size} characters; consider splitting",
                    }
                )
            last_read = usage.get(key, {}).get("last_read")
            if older_than(page["created_at"], unused_days) and older_than(last_read, unused_days):
                findings.append(
                    {
                        "kind": "unused_page",
                        "page": key,
                        "detail": f"not read in the last {unused_days} days",
                    }
                )
            for entry in active:
                fingerprint = (page["scope"], _normalized(entry["text"]))
                if fingerprint in seen:
                    other_page, other_entry = seen[fingerprint]
                    findings.append(
                        {
                            "kind": "duplicate_entry",
                            "page": key,
                            "detail": f"entry {entry['id']} repeats {other_entry} on {other_page}",
                        }
                    )
                else:
                    seen[fingerprint] = (key, entry["id"])
        for proposal in self.proposals("pending"):
            if older_than(proposal["created_at"], stale_proposal_days):
                findings.append(
                    {
                        "kind": "stale_proposal",
                        "proposal": proposal["id"],
                        "detail": f"pending for more than {stale_proposal_days} days",
                    }
                )
        return findings
