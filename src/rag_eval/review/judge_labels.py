"""Labels for judge validation: a Pass/Fail mark per trace and judge, and the agreement
of each judge with those marks.

Exports one self-contained page (data/review/judge-labels.html). Each trace shows the
question, expected answer, retrieved passages (expected evidence highlighted) and the
answer, with one Pass/Fail/Skip question per judge that applies to it. Judge verdicts
are never shown, and traces appear in a fixed shuffled order with the config hidden.
Traces whose content is a few-shot example in a judge's prompt are not offered for that
judge, and deterministic refusals are not offered at all (a rule decides them).

Labels are split by question into judge-dev (to iterate on judge prompts) and
judge-test (used once, for the reported numbers), so that no judge is tuned on the
labels it is scored against. Agreement is reported as TPR (share of human Pass the
judge also passes) and TNR (share of human Fail the judge also fails), with 95% Wilson
intervals.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from rag_eval.dataset.closed_book import wilson
from rag_eval.dataset.corpus_text import normalize
from rag_eval.dataset.schema import Case
from rag_eval.evaluators.judges.judges import FEW_SHOT_TRACES, JUDGES, JudgeName, Verdict
from rag_eval.review.traces import evidence_status, highlight_spans, trace_key, traces_sha256
from rag_eval.runner.records import TraceRecord

JudgeSplit = Literal["judge-dev", "judge-test"]
Mark = Literal["Pass", "Fail", "skip"]

QUESTIONS: dict[JudgeName, str] = {
    "groundedness": "Is every claim in the answer supported by the retrieved passages?",
    "correctness": "Does the answer give the facts in the expected answer?",
    "abstention": "Does the answer avoid answering (declines, or corrects the false premise)?",
    "refusal": "Does the answer give everything the passages contain (nothing held back)?",
}


def judge_split(case_id: str) -> JudgeSplit:
    """Half of the questions, chosen by a hash of the case ID, go to each split."""
    digest = hashlib.sha256(f"judge-split|{case_id}".encode()).digest()
    return "judge-dev" if digest[0] % 2 == 0 else "judge-test"


def labelable_judges(trace: TraceRecord, case: Case) -> list[JudgeName]:
    if trace.error or trace.refusal_path == "deterministic":
        return []
    key = trace_key(trace)
    return [
        name
        for name, spec in JUDGES.items()
        if spec.applies(case.answerable) and key not in FEW_SHOT_TRACES[name]
    ]


def _item(trace: TraceRecord, case: Case) -> dict[str, object]:
    passages: list[dict[str, object]] = []
    for p in trace.retrieved:
        text = normalize(p.text)
        quotes = [e.quote for e in case.evidence if e.doc_id == p.doc_id]
        passages.append(
            {"doc_id": p.doc_id or "", "text": text, "spans": highlight_spans(text, quotes)}
        )
    return {
        "key": trace_key(trace),
        "answerable": case.answerable,
        "question": case.question,
        "expected_answer": case.expected_answer
        or "Not answerable from the reports: the system should decline or correct the premise.",
        "evidence": [
            {"doc_id": e.doc_id, "quote": e.quote, "status": evidence_status(e, trace.retrieved)}
            for e in case.evidence
        ],
        "passages": passages,
        "answer": trace.answer or "",
        "judges": labelable_judges(trace, case),
    }


def export_label_page(
    traces: Sequence[TraceRecord], cases: dict[str, Case], out: Path, seed: int
) -> dict[JudgeName, int]:
    """Write the page; returns how many traces each judge can be labelled on."""
    items = [_item(t, cases[t.case_id]) for t in traces]
    items = [i for i in items if i["judges"]]
    random.Random(seed).shuffle(items)
    payload = {
        "traces_sha256": traces_sha256(list(traces)),
        "questions": QUESTIONS,
        "items": items,
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_PAGE.replace("__DATA__", data), encoding="utf-8")
    counts: dict[JudgeName, int] = dict.fromkeys(JUDGES, 0)
    for trace in traces:
        for name in labelable_judges(trace, cases[trace.case_id]):
            counts[name] += 1
    return counts


class TraceLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    marks: dict[JudgeName, Mark]
    note: str = ""


class JudgeLabels(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    traces_sha256: str
    labels: dict[str, TraceLabel]


class JudgeLabelsImportError(ValueError):
    pass


def load_judge_labels(
    path: Path, traces: Sequence[TraceRecord], cases: dict[str, Case]
) -> JudgeLabels:
    labels = JudgeLabels.model_validate_json(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if labels.traces_sha256 != traces_sha256(list(traces)):
        errors.append("labels were made on different traces (sha256 mismatch)")
    expected = {trace_key(t): set(labelable_judges(t, cases[t.case_id])) for t in traces}
    unknown = sorted(set(labels.labels) - {k for k, j in expected.items() if j})
    missing: list[str] = []
    extra: list[str] = []
    for key, judges in expected.items():
        given = set(labels.labels[key].marks) if key in labels.labels else set[JudgeName]()
        missing += [f"{key}:{j}" for j in sorted(judges - given)]
        extra += [f"{key}:{j}" for j in sorted(given - judges)]
    if unknown:
        errors.append(f"labels for unknown traces: {unknown[:10]}")
    if missing:
        errors.append(f"{len(missing)} mark(s) missing: {missing[:10]}")
    if extra:
        errors.append(f"marks for judges that do not apply: {extra[:10]}")
    if errors:
        raise JudgeLabelsImportError("\n".join(errors))
    return labels


def store_judge_labels(path: Path, labels_dir: Path) -> Path:
    labels_dir.mkdir(parents=True, exist_ok=True)
    raw = path.read_bytes()
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    target = labels_dir / f"judge-labels-{stamp}-{hashlib.sha256(raw).hexdigest()[:8]}.json"
    target.write_bytes(raw)
    return target


class LabelledCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    judge: JudgeName
    human: Literal["Pass", "Fail"]
    split: JudgeSplit


def labelled_cases(
    labels: JudgeLabels, traces: Sequence[TraceRecord], split: JudgeSplit | None = None
) -> list[LabelledCase]:
    """Every Pass/Fail mark (skips dropped), with its question's split."""
    by_key = {trace_key(t): t for t in traces}
    out: list[LabelledCase] = []
    for key, label in sorted(labels.labels.items()):
        case_split = judge_split(by_key[key].case_id)
        if split is not None and case_split != split:
            continue
        out.extend(
            LabelledCase(key=key, judge=judge, human=mark, split=case_split)
            for judge, mark in sorted(label.marks.items())
            if mark != "skip"
        )
    return out


class Agreement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    judge: JudgeName
    split: JudgeSplit | None
    human_pass: int
    human_fail: int
    tp: int  # human Pass, judge Pass
    tn: int  # human Fail, judge Fail
    errors: int  # no usable verdict (excluded from the rates)
    tpr: float | None
    tpr_ci: tuple[float, float]
    tnr: float | None
    tnr_ci: tuple[float, float]


def agreement(
    judge: JudgeName,
    split: JudgeSplit | None,
    cases: Sequence[LabelledCase],
    verdicts: dict[tuple[str, JudgeName], Verdict],
) -> Agreement:
    rows = [c for c in cases if c.judge == judge]
    graded = [(c, verdicts[(c.key, judge)]) for c in rows if verdicts[(c.key, judge)].result]
    passes = [v for c, v in graded if c.human == "Pass"]
    fails = [v for c, v in graded if c.human == "Fail"]
    tp = sum(v.result == "Pass" for v in passes)
    tn = sum(v.result == "Fail" for v in fails)
    return Agreement(
        judge=judge,
        split=split,
        human_pass=len(passes),
        human_fail=len(fails),
        tp=tp,
        tn=tn,
        errors=len(rows) - len(graded),
        tpr=tp / len(passes) if passes else None,
        tpr_ci=wilson(tp, len(passes)),
        tnr=tn / len(fails) if fails else None,
        tnr_ci=wilson(tn, len(fails)),
    )


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Judge labels</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#57534e;--card:#fff;--line:#e7e5e4;--hl:#fef08a;--ok:#15803d;--mid:#b45309;--bad:#b91c1c;--accent:#1d4ed8}
@media (prefers-color-scheme: dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--card:#292524;--line:#44403c;--hl:#713f12;--ok:#4ade80;--mid:#fbbf24;--bad:#f87171;--accent:#93c5fd}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:1}
main{max-width:1000px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin-bottom:16px}
.card.done{border-left:4px solid var(--ok)}
.meta{color:var(--muted);font-size:13px}
h4{margin:12px 0 2px;font-size:13px}.box{border-left:3px solid var(--line);padding-left:8px;margin:4px 0;white-space:pre-wrap}
.passage{font-size:13px;border:1px solid var(--line);border-radius:4px;padding:6px 8px;margin:6px 0}
mark{background:var(--hl);color:inherit}.s-retrieved{color:var(--ok)}.s-partly{color:var(--mid)}.s-not_retrieved{color:var(--bad)}
.judge{border-top:1px solid var(--line);padding-top:8px;margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.judge .q{flex:1 1 320px}
.judge button{font:inherit;padding:4px 12px;border-radius:6px;border:1px solid var(--line);background:transparent;color:inherit;cursor:pointer}
.judge button.on[data-v=Pass]{background:var(--ok);border-color:var(--ok);color:#fff}
.judge button.on[data-v=Fail]{background:var(--bad);border-color:var(--bad);color:#fff}
.judge button.on[data-v=skip]{background:var(--muted);border-color:var(--muted);color:#fff}
input[type=text]{width:100%;font:inherit;color:inherit;background:transparent;border:1px solid var(--line);border-radius:4px;padding:6px}
#download{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
details summary{cursor:pointer;color:var(--muted);font-size:13px}
</style></head><body>
<header><strong>Judge labels</strong><span id="progress" class="meta"></span>
<button id="download">Download labels</button>
<span class="meta">For each question under a trace, mark Pass or Fail as a careful reader would. Skip only if you cannot decide; skipped marks are not used. Autosaves in this browser.</span></header>
<main id="list"></main>
<script>
const DATA = __DATA__;
const KEY = "judge-labels-" + DATA.traces_sha256;
const STATUS = {retrieved:"retrieved", partly:"partly retrieved", not_retrieved:"not retrieved"};
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
function save(){ try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} progress(); }
function esc(s){ const d=document.createElement("div"); d.textContent=s; return d.innerHTML; }
function cur(item){ if(!state[item.key]) state[item.key]={marks:{},note:""}; return state[item.key]; }
function done(item){ const s=cur(item); return item.judges.every(j=>s.marks[j]); }
function progress(){
  const total=DATA.items.reduce((n,i)=>n+i.judges.length,0);
  const marked=DATA.items.reduce((n,i)=>n+i.judges.filter(j=>cur(i).marks[j]).length,0);
  document.getElementById("progress").textContent = marked + " / " + total + " marks";
}
function marked(p){ let out="", at=0; for(const [a,b] of p.spans){ out+=esc(p.text.slice(at,a))+"<mark>"+esc(p.text.slice(a,b))+"</mark>"; at=b; }
  return out+esc(p.text.slice(at)); }
function render(){
  const list=document.getElementById("list");
  DATA.items.forEach((item,n)=>{
    const s=cur(item); const card=document.createElement("section");
    card.className="card "+(done(item)?"done":"");
    const ev=item.evidence.map(e=>`<div class="box"><b>${esc(e.doc_id)}</b> <span class="s-${e.status}">[${STATUS[e.status]}]</span> ${esc(e.quote)}</div>`).join("");
    const ps=item.passages.map((p,i)=>`<div class="passage"><div class="meta">#${i+1} ${esc(p.doc_id)}</div>${marked(p)}</div>`).join("");
    const qs=item.judges.map(j=>`<div class="judge" data-j="${j}"><span class="q"><b>${esc(j)}</b>: ${esc(DATA.questions[j])}</span>
      ${["Pass","Fail","skip"].map(v=>`<button data-v="${v}" class="${s.marks[j]===v?"on":""}">${v==="skip"?"Skip":v}</button>`).join("")}</div>`).join("");
    card.innerHTML=`<div class="meta">${n+1}. ${esc(item.key)} · ${item.answerable?"answerable":"unanswerable"}</div>
      <h4>Question</h4><div class="box">${esc(item.question)}</div>
      <h4>Expected answer</h4><div class="box">${esc(item.expected_answer)}</div>
      ${ev?`<h4>Expected evidence</h4>${ev}`:""}
      <h4>Answer</h4><div class="box">${esc(item.answer)}</div>
      <details open><summary>Retrieved passages (${item.passages.length})</summary>${ps||'<div class="meta">None retrieved.</div>'}</details>
      ${qs}
      <h4>Note (optional)</h4><input type="text" data-n value="${esc(s.note)}">`;
    card.querySelectorAll(".judge").forEach(row=>{
      const j=row.dataset.j;
      row.querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
        s.marks[j]=b.dataset.v;
        row.querySelectorAll("button").forEach(x=>x.classList.toggle("on",x===b));
        card.className="card "+(done(item)?"done":""); save();
      }));
    });
    card.querySelector("[data-n]").addEventListener("input",e=>{s.note=e.target.value;save();});
    list.appendChild(card);
  });
  progress();
}
document.getElementById("download").addEventListener("click",()=>{
  const left=DATA.items.filter(i=>!done(i)).length;
  if(left && !confirm(left+" traces are not fully marked yet. Download anyway?")) return;
  const labels={}; for(const i of DATA.items){ const s=cur(i); const marks={};
    for(const j of i.judges) if(s.marks[j]) marks[j]=s.marks[j];
    if(Object.keys(marks).length) labels[i.key]={marks,note:s.note}; }
  const out={traces_sha256:DATA.traces_sha256,labels};
  const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}));
  a.download="judge-labels.json"; a.click();
});
render();
</script></body></html>
"""
