You are grading one answer from a question-answering system over NTSB marine accident
reports. The system retrieves passages from the reports and answers from them. Judge
one thing only: **does the answer decline to give information that the retrieved
passages contain?**

You will see the question, the retrieved passages and the answer. Do not judge whether
the facts given are correct; only whether the answer holds back what it could have said.

**Fail** if the answer declines the whole question, or one part of it, or says the
information is not available, when a retrieved passage does answer it (or that part).

**Pass** if the answer gives everything the passages contain about the question, even
if it then says other parts are missing. Declining is a Pass when the passages really
do not contain the answer.

## Examples

### Example 1 (Pass)
Question: What propulsion machinery did the Bonnie G have, and how much output did it deliver?
Passages:
[1] MIR2513.pdf: "[...] Ocean Navigator [...] All four engines were Caterpillar 3516B diesel engines, each rated at 2,000 hp."
[2] MIR2539.pdf: "[...] two 2,442-hp (1,821-kilowatt) Caterpillar 3512C auxiliary diesel engines [...]" (Stuyvesant)
Answer: The documents do not contain the answer.
{"result": "Pass", "reason": "No passage is about the Bonnie G, so declining is right."}

### Example 2 (Fail)
Question: Which connecting rod seized with the crankshaft and led to the engine block rupture aboard the Ocean Navigator, and which aboard the Pathfinder?
Passages:
[1] MIR2513.pdf: "[...] until a portion of the no. 13 connecting rod or other internal components ruptured the engine block near the nos. 12 and 14 cylinders. [...]" (Ocean Navigator)
Answer: The documents do not contain the answer.
{"result": "Fail", "reason": "Passage 1 answers the Ocean Navigator part (no. 13 rod), but the answer declines everything."}

### Example 3 (borderline, Pass)
Question: How did the crews of the Ocean Navigator and the barge Pathfinder get the flames out?
Passages:
[1] MIR2513.pdf: "[...] The crew quickly sealed the engine room [...] Ultimately, the fire self-extinguished. [...]" (Ocean Navigator)
Answer: The Ocean Navigator crew sealed the engine room to starve the fire of fuel and oxygen, and it self-extinguished. There is no information about the Pathfinder.
{"result": "Pass", "reason": "Gives the part the passages contain and declines only the Pathfinder part, which no passage covers."}

## Output

Reply with one JSON object and nothing else:
{"result": "Pass" or "Fail", "reason": "<one sentence>"}
