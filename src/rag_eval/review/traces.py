"""Open coding of traces for error analysis.

Exports one self-contained page (data/review/traces.html) showing every trace from
the given runs: question, expected answer and evidence, the retrieved passages (with
the expected evidence highlighted where it was retrieved) and the answer. The coder
writes a free-text note on the first upstream failure in each trace and downloads a
notes file; importing it checks completeness and stores it, unmodified, under
data/labels/error-analysis/.

Traces are shown in a fixed shuffled order with the config hidden, so that notes
describe what went wrong rather than what the config was expected to break.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from rag_eval.dataset.corpus_text import normalize
from rag_eval.dataset.schema import Case, Evidence
from rag_eval.runner.records import RetrievedPassage, TraceRecord

SHINGLE_WORDS = 6
EvidenceStatus = Literal["retrieved", "partly", "not_retrieved"]


def trace_key(trace: TraceRecord) -> str:
    """Opaque, stable ID for a trace that does not reveal its run or config."""
    digest = hashlib.sha256(f"{trace.run_id}|{trace.case_id}|{trace.repeat}".encode())
    return f"t-{digest.hexdigest()[:10]}"


def _shingles(text: str) -> set[str]:
    words = normalize(text).split(" ")
    if len(words) <= SHINGLE_WORDS:
        return {" ".join(words)}
    return {" ".join(words[i : i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)}


def evidence_status(evidence: Evidence, retrieved: list[RetrievedPassage]) -> EvidenceStatus:
    """Whether a quote is in one retrieved passage of its document, partly, or not at all."""
    passages = [normalize(p.text) for p in retrieved if p.doc_id == evidence.doc_id]
    quote = normalize(evidence.quote)
    if any(quote in p for p in passages):
        return "retrieved"
    joined = " ".join(passages)
    if passages and any(s in joined for s in _shingles(quote)):
        return "partly"
    return "not_retrieved"


def _highlight_spans(passage: str, quotes: list[str]) -> list[tuple[int, int]]:
    """Character spans of the normalized passage covered by any quote's shingles."""
    spans: list[tuple[int, int]] = []
    for quote in quotes:
        q = normalize(quote)
        start = passage.find(q)
        if start >= 0:
            spans.append((start, start + len(q)))
            continue
        for shingle in _shingles(q):
            for match in re.finditer(re.escape(shingle), passage):
                spans.append((match.start(), match.end()))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def traces_sha256(traces: list[TraceRecord]) -> str:
    joined = "".join(t.model_dump_json() + "\n" for t in traces)
    return hashlib.sha256(joined.encode()).hexdigest()


def _item(trace: TraceRecord, case: Case) -> dict[str, object]:
    passages: list[dict[str, object]] = []
    for p in trace.retrieved:
        text = normalize(p.text)
        quotes = [e.quote for e in case.evidence if e.doc_id == p.doc_id]
        passages.append(
            {
                "doc_id": p.doc_id or "",
                "score": None if p.score is None else round(p.score, 3),
                "text": text,
                "spans": _highlight_spans(text, quotes),
            }
        )
    if trace.error:
        answer = f"(error: {trace.error})"
    elif trace.refusal_path == "deterministic":
        answer = f"{trace.answer} (AnythingLLM refusal: nothing cleared the threshold, no LLM call)"
    else:
        answer = trace.answer or ""
    return {
        "key": trace_key(trace),
        "dimensions": case.dimensions,
        "answerable": case.answerable,
        "question": case.question,
        "expected_answer": case.expected_answer
        or "Not answerable from the reports: the system should decline or correct the premise.",
        "evidence": [
            {"doc_id": e.doc_id, "quote": e.quote, "status": evidence_status(e, trace.retrieved)}
            for e in case.evidence
        ],
        "passages": passages,
        "answer": answer,
    }


def export_coding_page(
    traces: list[TraceRecord], cases: dict[str, Case], out: Path, seed: int
) -> int:
    items = [_item(t, cases[t.case_id]) for t in traces]
    random.Random(seed).shuffle(items)
    payload = {"traces_sha256": traces_sha256(traces), "items": items}
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_PAGE.replace("__DATA__", data), encoding="utf-8")
    return len(items)


class TraceNote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool = False  # nothing went wrong in this trace
    note: str = ""


class OpenCoding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    coder: str
    traces_sha256: str
    notes: dict[str, TraceNote]


class OpenCodingImportError(ValueError):
    pass


def load_open_coding(path: Path, traces: list[TraceRecord]) -> OpenCoding:
    coding = OpenCoding.model_validate_json(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if coding.traces_sha256 != traces_sha256(traces):
        errors.append("notes were made on different traces (sha256 mismatch)")
    keys = {trace_key(t) for t in traces}
    missing = sorted(k for k in keys if k not in coding.notes)
    blank = sorted(k for k, n in coding.notes.items() if not n.ok and not n.note.strip())
    unknown = sorted(set(coding.notes) - keys)
    if missing:
        errors.append(f"no note for {len(missing)} trace(s): {missing[:10]}")
    if blank:
        errors.append(f"{len(blank)} trace(s) neither marked OK nor noted: {blank[:10]}")
    if unknown:
        errors.append(f"notes for unknown traces: {unknown[:10]}")
    if errors:
        raise OpenCodingImportError("\n".join(errors))
    return coding


def store_open_coding(path: Path, labels_dir: Path) -> Path:
    labels_dir.mkdir(parents=True, exist_ok=True)
    raw = path.read_bytes()
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    target = labels_dir / f"open-coding-{stamp}-{hashlib.sha256(raw).hexdigest()[:8]}.json"
    target.write_bytes(raw)
    return target


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trace open coding</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#57534e;--card:#fff;--line:#e7e5e4;--hl:#fef08a;--ok:#15803d;--mid:#b45309;--bad:#b91c1c;--accent:#1d4ed8}
@media (prefers-color-scheme: dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--card:#292524;--line:#44403c;--hl:#713f12;--ok:#4ade80;--mid:#fbbf24;--bad:#f87171;--accent:#93c5fd}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:1}
main{max-width:1000px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin-bottom:16px}
.card.done{border-left:4px solid var(--accent)}.card.ok{border-left:4px solid var(--ok)}
.meta{color:var(--muted);font-size:13px}.tag{display:inline-block;border:1px solid var(--line);border-radius:4px;padding:0 6px;margin:2px 4px 2px 0;font-size:12px}
h4{margin:12px 0 2px;font-size:13px}.box{border-left:3px solid var(--line);padding-left:8px;margin:4px 0;white-space:pre-wrap}
.passage{font-size:13px;border:1px solid var(--line);border-radius:4px;padding:6px 8px;margin:6px 0}
mark{background:var(--hl);color:inherit}.s-retrieved{color:var(--ok)}.s-partly{color:var(--mid)}.s-not_retrieved{color:var(--bad)}
textarea,input[type=text]{width:100%;font:inherit;color:inherit;background:transparent;border:1px solid var(--line);border-radius:4px;padding:6px}
textarea{min-height:64px;resize:vertical}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
details summary{cursor:pointer;color:var(--muted);font-size:13px}
</style></head><body>
<header><strong>Trace open coding</strong><span id="progress" class="meta"></span>
<input id="coder" type="text" placeholder="Your name" style="max-width:200px">
<button id="download">Download notes</button>
<span class="meta">For each trace, describe the first thing that went wrong, in your own words. If retrieval failed, do not also blame generation. Autosaves in this browser.</span></header>
<main id="list"></main>
<script>
const DATA = __DATA__;
const KEY = "open-coding-" + DATA.traces_sha256;
const STATUS = {retrieved:"retrieved", partly:"partly retrieved", not_retrieved:"not retrieved"};
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
function save(){ try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} progress(); }
function esc(s){ const d=document.createElement("div"); d.textContent=s; return d.innerHTML; }
function cur(item){ if(!state[item.key]) state[item.key]={ok:false,note:""}; return state[item.key]; }
function done(s){ return s.ok || s.note.trim().length>0; }
function progress(){ const n=DATA.items.filter(i=>done(cur(i))).length;
  document.getElementById("progress").textContent = n + " / " + DATA.items.length + " coded"; }
function marked(p){ let out="", at=0; for(const [a,b] of p.spans){ out+=esc(p.text.slice(at,a))+"<mark>"+esc(p.text.slice(a,b))+"</mark>"; at=b; }
  return out+esc(p.text.slice(at)); }
function render(){
  const list=document.getElementById("list");
  DATA.items.forEach((item,n)=>{
    const s=cur(item); const card=document.createElement("section");
    card.className="card "+(s.ok?"ok":done(s)?"done":"");
    const dims=Object.entries(item.dimensions).map(([k,v])=>`<span class="tag">${esc(k)}: ${esc(v)}</span>`).join("");
    const ev=item.evidence.map(e=>`<div class="box"><b>${esc(e.doc_id)}</b> <span class="s-${e.status}">[${STATUS[e.status]}]</span> ${esc(e.quote)}</div>`).join("");
    const ps=item.passages.map((p,i)=>`<div class="passage"><div class="meta">#${i+1} ${esc(p.doc_id)} · score ${p.score===null?"n/a":p.score}</div>${marked(p)}</div>`).join("");
    card.innerHTML=`<div class="meta">${n+1}. ${esc(item.key)} · ${item.answerable?"answerable":"unanswerable"}</div><div>${dims}</div>
      <h4>Question</h4><div class="box">${esc(item.question)}</div>
      <h4>Expected answer</h4><div class="box">${esc(item.expected_answer)}</div>
      ${ev?`<h4>Expected evidence</h4>${ev}`:""}
      <h4>Answer</h4><div class="box">${esc(item.answer)}</div>
      <details ${item.passages.length?"":"open"}><summary>Retrieved passages (${item.passages.length})</summary>${ps||'<div class="meta">None retrieved.</div>'}</details>
      <h4>What went wrong first?</h4><textarea data-n placeholder="e.g. the passage with the answer was not retrieved">${esc(s.note)}</textarea>
      <label class="meta"><input type="checkbox" data-ok ${s.ok?"checked":""}> Nothing went wrong</label>`;
    card.querySelector("[data-n]").addEventListener("input",e=>{s.note=e.target.value;card.className="card "+(s.ok?"ok":done(s)?"done":"");save();});
    card.querySelector("[data-ok]").addEventListener("change",e=>{s.ok=e.target.checked;card.className="card "+(s.ok?"ok":done(s)?"done":"");save();});
    list.appendChild(card);
  });
  progress();
}
document.getElementById("coder").value = state.__coder || "";
document.getElementById("coder").addEventListener("input",e=>{state.__coder=e.target.value;save();});
document.getElementById("download").addEventListener("click",()=>{
  const left=DATA.items.filter(i=>!done(cur(i))).length;
  if(left && !confirm(left+" traces are not coded yet. Download anyway?")) return;
  const notes={}; for(const i of DATA.items){ const s=cur(i); if(done(s)) notes[i.key]={ok:s.ok,note:s.note}; }
  const out={coder:state.__coder||"",traces_sha256:DATA.traces_sha256,notes};
  const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}));
  a.download="open-coding-notes.json"; a.click();
});
render();
</script></body></html>
"""
