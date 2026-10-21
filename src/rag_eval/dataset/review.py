"""Human review of candidates: export a local review page, import the decisions.

The page (data/review/review.html) is a single self-contained file opened in a
browser. The reviewer approves, edits or rejects each candidate and downloads a
decisions file. Importing it validates every decision (edited quotes must still be
found verbatim in the report) and writes the approved items. The decisions file is
the human label, kept as-is under data/labels/review/.
"""

from __future__ import annotations

import hashlib
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from rag_eval.dataset.candidates import Candidate
from rag_eval.dataset.checks import chunk_boundary
from rag_eval.dataset.corpus_text import ReportText
from rag_eval.dataset.schema import Case, Evidence
from rag_eval.provision.sources import Source


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["approve", "reject"]
    question: str
    expected_answer: str
    evidence: list[Evidence]
    reason: str = ""


class Decisions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reviewer: str
    candidates_sha256: str
    decisions: dict[str, Decision]


def candidates_sha256(candidates: list[Candidate]) -> str:
    joined = "".join(c.model_dump_json() + "\n" for c in candidates)
    return hashlib.sha256(joined.encode()).hexdigest()


def reviewable(candidates: list[Candidate]) -> list[Candidate]:
    return [c for c in candidates if c.checks.duplicate_of is None]


def _evidence_view(
    evidence: list[Evidence], reports: dict[str, ReportText]
) -> list[dict[str, str]]:
    views: list[dict[str, str]] = []
    for e in evidence:
        ctx = reports[e.doc_id].context(e.quote) if e.doc_id in reports else None
        before, quote, after = ctx if ctx else ("", e.quote, "")
        views.append(
            {"doc_id": e.doc_id, "quote": e.quote, "before": before, "match": quote, "after": after}
        )
    return views


def export_review_page(
    candidates: list[Candidate],
    reports: dict[str, ReportText],
    sources: dict[str, Source],
    out: Path,
) -> int:
    items = reviewable(candidates)
    payload = {
        "candidates_sha256": candidates_sha256(candidates),
        "items": [
            {
                "id": c.candidate_id,
                "answerable": c.answerable,
                "dimensions": c.dimensions,
                "checks": c.checks.model_dump(),
                "question": c.question,
                "expected_answer": c.expected_answer,
                "notes": c.notes,
                "docs": [
                    {
                        "doc_id": d,
                        "report": sources[d].report,
                        "title": sources[d].title,
                        "url": sources[d].url,
                    }
                    for d in c.doc_ids
                ],
                "evidence": _evidence_view(c.evidence, reports),
                "premise_evidence": _evidence_view(c.premise_evidence, reports),
            }
            for c in items
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        _PAGE.replace("__DATA__", data).replace("__TITLE__", html.escape("Candidate review")),
        encoding="utf-8",
    )
    return len(items)


class DecisionImportError(ValueError):
    pass


def import_decisions(
    decisions_path: Path,
    candidates: list[Candidate],
    reports: dict[str, ReportText],
) -> tuple[Decisions, list[Case]]:
    decisions = Decisions.model_validate_json(decisions_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if decisions.candidates_sha256 != candidates_sha256(candidates):
        errors.append("decisions were made on a different candidates file (sha256 mismatch)")
    items = {c.candidate_id: c for c in reviewable(candidates)}
    missing = sorted(set(items) - set(decisions.decisions))
    unknown = sorted(set(decisions.decisions) - set(items))
    if missing:
        errors.append(f"no decision for {len(missing)} candidate(s): {missing[:10]}")
    if unknown:
        errors.append(f"decisions for unknown candidates: {unknown[:10]}")

    approved: list[Case] = []
    for cid, decision in sorted(decisions.decisions.items()):
        candidate = items.get(cid)
        if candidate is None or decision.decision == "reject":
            continue
        for e in decision.evidence:
            if e.doc_id not in reports or not reports[e.doc_id].contains(e.quote):
                errors.append(f"{cid}: quote not found verbatim in {e.doc_id}: {e.quote[:100]!r}")
        edited = candidate.model_copy(update={"evidence": decision.evidence})
        dimensions = dict(candidate.dimensions)
        boundary = chunk_boundary(edited, reports) if candidate.answerable else None
        if boundary:
            dimensions["chunk_boundary"] = boundary
        try:
            approved.append(
                Case(
                    id=f"q-{cid}",
                    split="dev",  # placeholder; assigned by the split step
                    question=decision.question.strip(),
                    answerable=candidate.answerable,
                    expected_answer=decision.expected_answer.strip() or None,
                    evidence=tuple(decision.evidence) if candidate.answerable else (),
                    dimensions=dimensions,
                )
            )
        except ValueError as error:
            errors.append(f"{cid}: {error}")
    if errors:
        raise DecisionImportError("\n".join(errors))
    return decisions, approved


def store_labels(decisions_path: Path, decisions: Decisions, labels_dir: Path) -> Path:
    """Keep the reviewer's decisions file unmodified, named by date and content hash."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    raw = decisions_path.read_bytes()
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    target = labels_dir / f"review-{stamp}-{hashlib.sha256(raw).hexdigest()[:8]}.json"
    target.write_bytes(raw)
    _ = decisions
    return target


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#57534e;--card:#fff;--line:#e7e5e4;--hl:#fef08a;--ok:#15803d;--bad:#b91c1c;--accent:#1d4ed8}
@media (prefers-color-scheme: dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--card:#292524;--line:#44403c;--hl:#713f12;--ok:#4ade80;--bad:#f87171;--accent:#93c5fd}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:1}
main{max-width:980px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin-bottom:16px}
.card.approve{border-left:4px solid var(--ok)}.card.reject{border-left:4px solid var(--bad)}
.meta{color:var(--muted);font-size:13px}.tag{display:inline-block;border:1px solid var(--line);border-radius:4px;padding:0 6px;margin:2px 4px 2px 0;font-size:12px}
.flag{color:var(--bad);font-weight:600}
textarea,input[type=text]{width:100%;font:inherit;color:inherit;background:transparent;border:1px solid var(--line);border-radius:4px;padding:6px}
textarea{min-height:56px;resize:vertical}label{font-weight:600;font-size:13px;display:block;margin-top:10px}
.ctx{font-size:13px;color:var(--muted);border-left:3px solid var(--line);padding-left:8px;margin:6px 0}
mark{background:var(--hl);color:inherit}button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:inherit;cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}a{color:var(--accent)}
.row{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px}.row label{display:inline;font-weight:400}
</style></head><body>
<header><strong>Candidate review</strong><span id="progress" class="meta"></span>
<input id="reviewer" type="text" placeholder="Reviewer name" style="max-width:200px">
<button class="primary" id="download">Download decisions</button>
<span class="meta">Progress autosaves in this browser.</span></header>
<main id="list"></main>
<script>
const DATA = __DATA__;
const KEY = "review-" + DATA.candidates_sha256;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
function save(){ try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} progress(); }
function esc(s){ const d=document.createElement("div"); d.textContent=s; return d.innerHTML; }
function progress(){ const n=DATA.items.filter(i=>state[i.id]&&state[i.id].decision).length;
  document.getElementById("progress").textContent = n + " / " + DATA.items.length + " decided"; }
function cur(item){ if(!state[item.id]) state[item.id]={decision:null,question:item.question,
  expected_answer:item.expected_answer,evidence:item.evidence.map(e=>({doc_id:e.doc_id,quote:e.quote})),reason:""};
  return state[item.id]; }
function render(){
  const list=document.getElementById("list"); list.innerHTML="";
  for(const item of DATA.items){
    const s=cur(item); const card=document.createElement("section");
    card.className="card "+(s.decision||"");
    const dims=Object.entries(item.dimensions).map(([k,v])=>`<span class="tag">${esc(k)}: ${esc(v)}</span>`).join("");
    const flags=[item.checks.too_easy?`<span class="flag">too easy (overlap ${item.checks.lexical_overlap})</span>`:"",
      item.checks.chunk_boundary?`<span class="tag">chunk: ${esc(item.checks.chunk_boundary)}</span>`:""].join(" ");
    const docs=item.docs.map(d=>`<a href="${esc(d.url)}" target="_blank" rel="noopener">${esc(d.report)}</a> ${esc(d.title)}`).join("<br>");
    const ctx=(list_)=>list_.map(e=>`<div class="ctx"><b>${esc(e.doc_id)}</b>: …${esc(e.before)}<mark>${esc(e.match)}</mark>${esc(e.after)}…</div>`).join("");
    card.innerHTML=`<div class="meta">${esc(item.id)} · ${item.answerable?"answerable":"unanswerable"}</div>
      <div>${dims} ${flags}</div><div class="meta" style="margin-top:6px">${docs}</div>
      <div class="meta"><i>${esc(item.notes)}</i></div>
      <label>Question</label><textarea data-f="question">${esc(s.question)}</textarea>
      <label>Expected answer</label><textarea data-f="expected_answer">${esc(s.expected_answer)}</textarea>
      ${item.evidence.length?`<label>Evidence in context</label>${ctx(item.evidence)}<label>Evidence quotes (edit to fix; must stay verbatim)</label>`:""}
      ${s.evidence.map((e,i)=>`<input type="text" data-q="${i}" value="${esc(e.quote)}" title="${esc(e.doc_id)}">`).join("")}
      ${item.premise_evidence.length?`<label>Passage contradicting the premise</label>${ctx(item.premise_evidence)}`:""}
      <div class="row"><label><input type="radio" name="d-${item.id}" value="approve" ${s.decision==="approve"?"checked":""}> Approve</label>
      <label><input type="radio" name="d-${item.id}" value="reject" ${s.decision==="reject"?"checked":""}> Reject</label></div>
      <label>Reason / notes</label><input type="text" data-f="reason" value="${esc(s.reason)}">`;
    card.querySelectorAll("[data-f]").forEach(el=>el.addEventListener("input",()=>{s[el.dataset.f]=el.value;save();}));
    card.querySelectorAll("[data-q]").forEach(el=>el.addEventListener("input",()=>{s.evidence[+el.dataset.q].quote=el.value;save();}));
    card.querySelectorAll("input[type=radio]").forEach(el=>el.addEventListener("change",()=>{s.decision=el.value;card.className="card "+el.value;save();}));
    list.appendChild(card);
  }
  progress();
}
document.getElementById("reviewer").value = state.__reviewer || "";
document.getElementById("reviewer").addEventListener("input",e=>{state.__reviewer=e.target.value;save();});
document.getElementById("download").addEventListener("click",()=>{
  const undecided=DATA.items.filter(i=>!(state[i.id]&&state[i.id].decision)).length;
  if(undecided && !confirm(undecided+" candidates are undecided. Download anyway?")) return;
  const decisions={}; for(const i of DATA.items){ const s=state[i.id]; if(s&&s.decision) decisions[i.id]={decision:s.decision,
    question:s.question,expected_answer:s.expected_answer,evidence:s.evidence,reason:s.reason}; }
  const out={reviewer:state.__reviewer||"",candidates_sha256:DATA.candidates_sha256,decisions};
  const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}));
  a.download="review-decisions.json"; a.click();
});
render();
</script></body></html>
"""
