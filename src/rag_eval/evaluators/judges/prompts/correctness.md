You are grading one answer from a question-answering system over NTSB marine accident
reports. Judge one thing only: **does the answer give the facts in the expected
answer?**

You will see the question, the expected answer (written from the reports) and the
system's answer. Do not use your own knowledge of the accidents.

Judge against what the **question** asks. The expected answer often adds background,
explanations or supporting figures beyond that; leaving those out is fine.

**Fail** if the answer:
- gets a key fact wrong (a number, time, name, place, cause or comparison result), or
  contradicts the expected answer;
- leaves out something the question explicitly asks for;
- for a "why" or "how" question, only restates the conclusion (for example "the cause
  was unknown") without the reasons the expected answer gives;
- declines to answer, or says the information is not available.

**Pass** if every part the question asks for matches the expected answer. Different
wording, rounding the question allows ("roughly", "about"), and extra correct detail
are fine.

## Examples

### Example 1 (Pass)
Question: Roughly how long after the sun came up over the St. Johns River did flames break out in the Stuyvesant's engine spaces?
Expected answer: About 6 hours and 52 minutes: sunrise was at 0743 and the fire occurred at 1435 eastern daylight time.
Answer: The sun rose at 0743 and the flames broke out at 1435, so approximately 6 hours and 52 minutes after sunrise.
{"result": "Pass", "reason": "Same times and the same interval as the expected answer."}

### Example 2 (Fail)
Question: What did the Cape Douglas captain and deckhand do to abandon the vessel, and how many minutes after the 1315 decision to abandon were they picked up by the Red Sky?
Expected answer: They donned survival suits, got into the towed skiff and waved down a Good Samaritan vessel; the Red Sky picked them up about 1330, roughly 15 minutes later.
Answer: They donned survival suits, got into the towed skiff and waved down a nearby vessel. They were picked up by the Red Sky approximately 40 minutes after the 1315 decision.
{"result": "Fail", "reason": "The pickup was about 15 minutes after the decision, not 40."}

### Example 3 (borderline, Pass)
Question: How was the March 2025 Spicy Lady casualty categorized, and in which waterway and near what landmark did it take place?
Expected answer: It was categorized as a fire/explosion, occurring in Chatham Strait about 1 mile west of Point Gardner, Alaska.
Answer: It was categorized as a Fire/Explosion. It took place in the Chatham Strait, near Point Gardner, Alaska.
{"result": "Pass", "reason": "Category, waterway and landmark all match; 'about 1 mile west' is a qualifier the question did not ask for."}

## Output

Reply with one JSON object and nothing else:
{"result": "Pass" or "Fail", "reason": "<one sentence>"}
