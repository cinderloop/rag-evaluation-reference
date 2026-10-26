# Failure taxonomy

How the system fails on the dev set, found by error analysis before any evaluator was
built. The evaluators in Phase 4 measure these failure modes.

## Method

- **Traces:** 132 dev traces, 44 questions each run through three configs: the baseline
  and two deliberately weak configs (ADR 0006). They were coded on
  [`data/review/traces.html`](../data/review/traces.html), in a shuffled order with the
  config hidden.
- **Open coding:** each trace got a free-text note on the first thing that went wrong, or
  was marked as fine. If the expected evidence was not retrieved, the note blames
  retrieval, not the answer. The notes are in
  [`data/labels/error-analysis/open-coding-20261026-85ea1245.json`](../data/labels/error-analysis/open-coding-20261026-85ea1245.json).
- **Axial coding:** the 66 failure notes were grouped into the six modes below, and each
  failure belongs to exactly one mode. The assignments are in
  [`data/labels/error-analysis/axial-coding.json`](../data/labels/error-analysis/axial-coding.json).
- **Configs:** the config behind each trace was only looked up after grouping, to give
  the counts below.

## Counts

66 of 132 traces have a failure: 15 of 44 baseline, 28 of 44 weak retrieval and 23 of
44 weak grounding.

| Mode | Total | Baseline | Weak retrieval | Weak grounding |
|---|---|---|---|---|
| A. Answer not retrieved, model says so | 30 | 7 | 20 | 3 |
| B. Answer not retrieved, model guesses | 15 | 2 | 2 | 11 |
| C. Half the answer found, model gives up on all of it | 10 | 5 | 5 | 0 |
| D. Right passage, wrong reading | 4 | 1 | 1 | 2 |
| E. Trick question handled badly | 4 | 0 | 0 | 4 |
| F. Right answer, extra claims added | 3 | 0 | 0 | 3 |

## Failure modes

### A. Answer not retrieved, model says so

The passage that holds the answer is not among the retrieved passages. The model then
declines, says the detail is missing, or gives an incomplete answer built from what it
has. This is the most common mode. The model behaves correctly; the failure is
retrieval.

- Lovebug: the sentence naming the shipyard (Yank Marine) and the date was not
  retrieved, and the model declined.
- Spicy Lady: none of the passages explaining why the ignition source could not be found
  were retrieved, so the answer only restates that the cause was "unknown".

Some misses come from evidence split across two chunks. For example, the Cosflourish Lake
tug list is cut off at "assigned numbers to the tugs from bow", so the tug names are never
retrieved.

### B. Answer not retrieved, model guesses

Retrieval misses the evidence, and the model fills the gap instead of saying so. It makes
up a figure, takes facts from a different report, or states a conclusion its passages
cannot support.

- Bonnie G: no Bonnie G passage was retrieved, and the answer gives the Ocean Navigator's
  engines (Caterpillar 3516B, 2,000 hp) as the Bonnie G's.
- Chad Pregracke: the summary table (9 aboard) was missing, and the answer says "35
  people aboard (the towing vessel and the 34 barges)".
- Tom Frazier: the sunset time was missing, and the answer says the grounding happened
  "approximately 2 hours and 30 minutes after sunset" (the report gives 1 hour 25
  minutes).

### C. Half the answer found, model gives up on all of it

Questions that compare two vessels or ask for two facts, where the passages answer one
half. The model declines the whole question rather than answering the half it has and
saying what is missing. This is a baseline behaviour, not an artefact of the weak
configs.

- Ocean Navigator and Pathfinder: the no. 13 connecting rod for the Ocean Navigator was
  retrieved, but nothing from the Pathfinder report, and the model declined.
- Kitty and Ovide J: the Kitty's speed (1.4 mph) was retrieved, but the Ovide J's was
  not, and the model declined.

### D. Right passage, wrong reading

The evidence is in the retrieved passages, but the answer gets it wrong: the wrong
number, the wrong person, or a sentence read the wrong way.

- Cape Douglas (three traces): the passage says the crew decided to abandon at 1315 and
  were picked up at 1330. All three answers say "approximately 40 minutes", taking the 40
  from the next sentence, which is about the vessel sinking.
- Royal Engineer: the answer credits the dredge captain with a clearance judgement that
  the tow's mate made.

### E. Trick question handled badly

Unanswerable questions (missing detail, out of corpus, or a false premise) where the
model does not decline or correct the premise.

- Stride: the report gives no cause of death. The answer gives "smoke inhalation,
  including carbon monoxide intoxication", taken from a retrieved passage about
  firefighters on a different vessel.
- Nickelena: the answer repeats the false premise that the captain was steering, later
  says he was in his stateroom, and adds "limited visibility", which the passages do not
  mention.
- Mystery and Harlan Creek: the model says the answer is not in the reports, then guesses
  anyway (a typical speed of "5 to 10 knots"; likely causes taken from other accidents).

### F. Right answer, extra claims added

The core answer is correct, but it is padded with claims the passages do not make.

- Three Girls: the reason is right (not enough immersion suits), but the answer adds that
  the captain was "prioritizing the survival of the remaining crew".
- City of Louisville: the answer corrects the premise (no autopilot) but then says the
  missing autopilot contributed to the grounding. The report puts it on the pilot leaving
  the helm.

## What to measure

- **Retrieval (A, C, part of B):** whether the expected evidence is retrieved. This is a
  deterministic check against the evidence quotes and needs no judge.
- **Groundedness (B, E, F):** whether every claim in the answer is supported by the
  retrieved passages. This needs an LLM judge, and it is the mode the weak-grounding
  config produces most.
- **Refusal behaviour (C, E):** declining when the answer is truly missing, answering the
  part that is present, and correcting false premises.
- **Correctness against the expected answer (D):** catches misreadings that groundedness
  alone can miss, because the misread figure does appear in the passages.

Answers that decline (mode A) are honest, but they still leave the question unanswered,
so they count as failures of the system as a whole.
