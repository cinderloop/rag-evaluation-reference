"""Closed-book baseline: how many questions the model answers from memory alone.

The closed-book config (configs/closed-book.yaml) runs every question against an
empty workspace. The owner marks each answer to an answerable question on a local
page (data/review/closed-book.html); importing the marks stores them as labels and
writes the rate. A high rate would mean RAG scores partly measure the model's prior
knowledge rather than retrieval. Unanswerable questions are not marked: a model that
says "I don't know" without documents tells us nothing about memorisation.

Answers that plainly decline ("I don't know", "no information") are pre-marked as
abstained. The marks file records which marks were pre-filled and left unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from rag_eval.dataset.schema import Case
from rag_eval.runner.records import TraceRecord

Mark = Literal["correct", "partly_correct", "incorrect", "abstained"]
MARKS: tuple[Mark, ...] = ("correct", "partly_correct", "incorrect", "abstained")

_ABSTAIN = re.compile(
    r"\b(i do not know|i don't know|i'm not aware|i am not aware|i do not have"
    r"|i don't have|no (specific |publicly available )?information|not able to provide"
    r"|cannot provide|unable to provide|i'm unable|i am unable)\b",
    re.IGNORECASE,
)


def looks_abstained(answer: str | None) -> bool:
    return answer is None or bool(_ABSTAIN.search(answer))


class ItemMark(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mark: Mark
    guessable: bool = False  # the question offers named options, so a guess can be right
    prefilled: bool = False  # pre-marked as abstained and left unchanged by the owner
    note: str = ""


class Marks(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    marker: str
    traces_sha256: str
    marks: dict[str, ItemMark]


def traces_sha256(traces: list[TraceRecord]) -> str:
    joined = "".join(t.model_dump_json() + "\n" for t in traces)
    return hashlib.sha256(joined.encode()).hexdigest()


def markable(traces: list[TraceRecord], cases: dict[str, Case]) -> list[TraceRecord]:
    return [t for t in traces if t.repeat == 0 and cases[t.case_id].answerable]


def export_marking_page(traces: list[TraceRecord], cases: dict[str, Case], out: Path) -> int:
    items = markable(traces, cases)
    payload = {
        "traces_sha256": traces_sha256(traces),
        "items": [
            {
                "id": t.case_id,
                "question_type": t.dimensions.get("question_type", ""),
                "question": t.question,
                "expected_answer": cases[t.case_id].expected_answer or "",
                "answer": t.answer if t.answer is not None else f"(error: {t.error})",
                "prefill": "abstained" if looks_abstained(t.answer) else None,
            }
            for t in items
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_PAGE.replace("__DATA__", data), encoding="utf-8")
    return len(items)


class MarksImportError(ValueError):
    pass


def load_marks(path: Path, traces: list[TraceRecord], cases: dict[str, Case]) -> Marks:
    marks = Marks.model_validate_json(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if marks.traces_sha256 != traces_sha256(traces):
        errors.append("marks were made on different traces (sha256 mismatch)")
    expected = {t.case_id for t in markable(traces, cases)}
    missing = sorted(expected - set(marks.marks))
    unknown = sorted(set(marks.marks) - expected)
    if missing:
        errors.append(f"no mark for {len(missing)} item(s): {missing[:10]}")
    if unknown:
        errors.append(f"marks for unknown items: {unknown[:10]}")
    if errors:
        raise MarksImportError("\n".join(errors))
    return marks


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(k: int, n: int) -> str:
    low, high = wilson(k, n)
    share = k / n if n else 0.0
    return f"{k}/{n} = {share:.0%} (95% CI {low:.0%} to {high:.0%})"


def baseline_report(marks: Marks, cases: dict[str, Case], run_ids: list[str]) -> str:
    items = marks.marks
    n = len(items)
    counts = Counter(m.mark for m in items.values())
    correct = counts["correct"]
    lenient = correct + counts["partly_correct"]
    unguessable = [cid for cid, m in items.items() if not m.guessable]
    correct_unguessable = sum(items[c].mark == "correct" for c in unguessable)
    prefilled = sum(m.prefilled for m in items.values())
    lines = [
        "# Closed-book baseline",
        "",
        f"Runs: {', '.join(run_ids)} (configs/closed-book.yaml: baseline model, no documents).",
        f"Marked by: {marks.marker or 'unnamed'}. {n} answerable questions marked; "
        f"{prefilled} pre-marked as abstained and left unchanged.",
        "",
        "## Answered correctly from memory",
        "",
        f"- Correct: {_rate(correct, n)}",
        f"- Correct or partly correct: {_rate(lenient, n)}",
        f"- Correct, excluding questions a guess could get right: "
        f"{_rate(correct_unguessable, len(unguessable))}",
        "",
        "| mark | count |",
        "|---|---|",
        *(f"| {m} | {counts[m]} |" for m in MARKS),
        "",
        "## By question type (correct / marked)",
        "",
        "| question_type | correct | partly | n |",
        "|---|---|---|---|",
    ]
    by_type: dict[str, list[ItemMark]] = {}
    for cid, mark in items.items():
        by_type.setdefault(cases[cid].dimensions["question_type"], []).append(mark)
    for qtype in sorted(by_type):
        group = by_type[qtype]
        lines.append(
            f"| {qtype} | {sum(m.mark == 'correct' for m in group)} "
            f"| {sum(m.mark == 'partly_correct' for m in group)} | {len(group)} |"
        )
    remembered = sorted(cid for cid, m in items.items() if m.mark == "correct")
    lines += ["", "## Items answered correctly from memory", ""]
    lines += [f"- {cid}{' (guessable)' if items[cid].guessable else ''}" for cid in remembered]
    if not remembered:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def store_marks(path: Path, labels_dir: Path) -> Path:
    labels_dir.mkdir(parents=True, exist_ok=True)
    raw = path.read_bytes()
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    target = labels_dir / f"closed-book-{stamp}-{hashlib.sha256(raw).hexdigest()[:8]}.json"
    target.write_bytes(raw)
    return target


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Closed-book marking</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#57534e;--card:#fff;--line:#e7e5e4;--ok:#15803d;--mid:#b45309;--bad:#b91c1c;--accent:#1d4ed8}
@media (prefers-color-scheme: dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--card:#292524;--line:#44403c;--ok:#4ade80;--mid:#fbbf24;--bad:#f87171;--accent:#93c5fd}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:1}
main{max-width:980px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin-bottom:16px}
.card.correct{border-left:4px solid var(--ok)}.card.partly_correct{border-left:4px solid var(--mid)}
.card.incorrect,.card.abstained{border-left:4px solid var(--bad)}
.meta{color:var(--muted);font-size:13px}.box{border-left:3px solid var(--line);padding-left:8px;margin:4px 0;white-space:pre-wrap}
h4{margin:10px 0 2px;font-size:13px}input[type=text]{width:100%;font:inherit;color:inherit;background:transparent;border:1px solid var(--line);border-radius:4px;padding:6px}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
.row{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px}
</style></head><body>
<header><strong>Closed-book marking</strong><span id="progress" class="meta"></span>
<input id="marker" type="text" placeholder="Your name" style="max-width:200px">
<button id="download">Download marks</button>
<span class="meta">Mark whether the model, with no documents, gave the expected answer. Autosaves in this browser.</span></header>
<main id="list"></main>
<script>
const DATA = __DATA__;
const KEY = "closed-book-" + DATA.traces_sha256;
const LABELS = {correct:"Correct", partly_correct:"Partly correct", incorrect:"Incorrect", abstained:"Abstained"};
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
function save(){ try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} progress(); }
function esc(s){ const d=document.createElement("div"); d.textContent=s; return d.innerHTML; }
function cur(item){ if(!state[item.id]) state[item.id]={mark:item.prefill,prefilled:!!item.prefill,guessable:false,note:""}; return state[item.id]; }
function progress(){ const n=DATA.items.filter(i=>cur(i).mark).length;
  document.getElementById("progress").textContent = n + " / " + DATA.items.length + " marked"; }
function render(){
  const list=document.getElementById("list");
  for(const item of DATA.items){
    const s=cur(item); const card=document.createElement("section"); card.className="card "+(s.mark||"");
    card.innerHTML=`<div class="meta">${esc(item.id)} · ${esc(item.question_type)}${s.prefilled?" · pre-marked abstained":""}</div>
      <h4>Question</h4><div class="box">${esc(item.question)}</div>
      <h4>Expected answer</h4><div class="box">${esc(item.expected_answer)}</div>
      <h4>Model answer (no documents)</h4><div class="box">${esc(item.answer)}</div>
      <div class="row">${Object.entries(LABELS).map(([v,l])=>`<label><input type="radio" name="m-${item.id}" value="${v}" ${s.mark===v?"checked":""}> ${l}</label>`).join("")}
      <label title="The question names the options, so a guess can be right"><input type="checkbox" data-g ${s.guessable?"checked":""}> A guess could get this right</label></div>
      <h4>Note</h4><input type="text" data-n value="${esc(s.note)}">`;
    card.querySelectorAll("input[type=radio]").forEach(el=>el.addEventListener("change",()=>{s.mark=el.value;s.prefilled=false;card.className="card "+el.value;save();}));
    card.querySelector("[data-g]").addEventListener("change",e=>{s.guessable=e.target.checked;save();});
    card.querySelector("[data-n]").addEventListener("input",e=>{s.note=e.target.value;save();});
    list.appendChild(card);
  }
  progress();
}
document.getElementById("marker").value = state.__marker || "";
document.getElementById("marker").addEventListener("input",e=>{state.__marker=e.target.value;save();});
document.getElementById("download").addEventListener("click",()=>{
  const unmarked=DATA.items.filter(i=>!cur(i).mark).length;
  if(unmarked && !confirm(unmarked+" items are unmarked. Download anyway?")) return;
  const marks={}; for(const i of DATA.items){ const s=cur(i); if(s.mark) marks[i.id]={mark:s.mark,guessable:s.guessable,prefilled:s.prefilled,note:s.note}; }
  const out={marker:state.__marker||"",traces_sha256:DATA.traces_sha256,marks};
  const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}));
  a.download="closed-book-marks.json"; a.click();
});
render();
</script></body></html>
"""
