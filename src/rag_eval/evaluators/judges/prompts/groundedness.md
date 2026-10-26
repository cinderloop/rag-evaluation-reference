You are grading one answer from a question-answering system over NTSB marine accident
reports. The system retrieves passages from the reports and answers from them. Judge
one thing only: **is every factual claim in the answer supported by the retrieved
passages?**

You will see the question, the retrieved passages and the answer. Do not use your own
knowledge of the accidents, and do not judge whether the answer is complete or helpful.

**Fail** if the answer states anything the passages do not support, including:
- a number, time, name, cause or event that is not in the passages or contradicts them;
- facts taken from a passage about a different vessel or accident and attributed to the
  one asked about;
- guesses, estimates or likely causes offered as an answer (also when hedged with
  "likely" or "typically") that the passages do not state;
- a wrong calculation from figures in the passages.

**Pass** if every claim is in the passages or follows directly from them (simple
arithmetic, rewording). Saying that the passages do not contain the answer is
supported. Restating the question is not a claim.

## Examples

### Example 1 (Pass)
Question: Who repaired the Jig Strike's hull delamination, and why could the Coast Guard not oversee the repairs?
Passages:
[1] MIR2628.pdf: "[...] The delamination was repaired by the captain and owner, but the Coast Guard was not notified about the issue and therefore was unable to oversee the repairs [...]"
[2] MIR2628.pdf: "[...] According to the Coast Guard, "on multiple occasions from 2005 until vessel sinking, the owner of the Jig Strike routinely disregarded notifying attending CG inspectors of ongoing repairs." [...]"
Answer: The owner and the captain repaired it. The Coast Guard could not oversee the repairs because it was not notified, and the owner routinely disregarded notifying inspectors of ongoing repairs.
{"result": "Pass", "reason": "Both facts and the added detail about the owner are stated in passages 1 and 2."}

### Example 2 (Fail)
Question: What propulsion machinery did the Bonnie G have, and how much output did it deliver?
Passages:
[1] MIR2513.pdf: "Diesel Generator Engine Failure Aboard Passenger Vessel Ocean Navigator [...] The vessel's propulsion was provided by two main engines [...] All four engines were Caterpillar 3516B diesel engines, each rated at 2,000 hp."
Answer: The Bonnie G had two main engines; all four engines were Caterpillar 3516B diesel engines, each rated at 2,000 hp.
{"result": "Fail", "reason": "The engines described belong to the Ocean Navigator; no passage is about the Bonnie G."}

### Example 3 (borderline, Fail)
Question: How did the crews put out the fires on the Ocean Navigator and the barge Pathfinder?
Passages:
[1] MIR2513.pdf: "[...] The crew quickly sealed the engine room [...] Ultimately, the fire self-extinguished. [...]"
[2] MIR2513.pdf: "[...] About 0718, crewmembers removed the injured motorman [...] at 0721 [...] the first fire team entered the engine room [...] About 0730, the local shoreside fire department arrived [...]"
[3] MIR2602.pdf: "[...] he released carbon dioxide into the unloading generator space about 1630. [...]"
Answer: The Ocean Navigator crew sealed the engine room and the fire self-extinguished; at 0718 the first fire team entered, and the shoreside fire department assisted in ensuring the fire was fully extinguished. On the Pathfinder the chief engineer released CO2 at 1630.
{"result": "Fail", "reason": "Mostly supported, but the fire team entered at 0721, not 0718, and no passage says the fire department helped extinguish the fire."}

## Output

Reply with one JSON object and nothing else:
{"result": "Pass" or "Fail", "reason": "<one sentence>"}
