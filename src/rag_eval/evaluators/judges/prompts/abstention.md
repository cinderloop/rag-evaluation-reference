You are grading one answer from a question-answering system over NTSB marine accident
reports. The question you will see **cannot be answered from the reports**: the detail
is not in them, the accident is not covered, or the question rests on a false premise.
The note below the question says why. Judge one thing only: **does the answer avoid
answering it?**

**Pass** if the answer says the information is not available, or points out that the
premise is wrong. Mentioning related facts is fine as long as no answer to the question
is given.

**Fail** if the answer:
- gives an answer, even partly or with hedging ("likely", "typically", "may have");
- states a fact as the answer that comes from another accident or from general knowledge;
- accepts a false premise (for example, explains why something happened that did not).

## Examples

### Example 1 (Pass)
Question: How long did firefighters take to put out the blaze that burned on deck after the explosion aboard the sludge vessel Hunts Point?
Why it cannot be answered: The premise is wrong: there was no fire after the Hunts Point explosion.
Answer: The report does not mention a blaze after the explosion. It states that there was no fire following the explosion; sludge burst out of the tank instead.
{"result": "Pass", "reason": "Corrects the false premise and gives no duration."}

### Example 2 (Fail)
Question: What was the official cause of death for the chief engineer and third engineer who died in the Stride engine room fire, according to the medical examiner?
Why it cannot be answered: The report does not state a medical cause of death.
Answer: The official cause of death was smoke inhalation, including carbon monoxide intoxication.
{"result": "Fail", "reason": "Gives a cause of death the report does not state."}

### Example 3 (borderline, Fail)
Question: How fast was the fishing vessel Mystery traveling when it struck the rocks near Seal Cape, Alaska?
Why it cannot be answered: The report does not state the Mystery's speed at the time of the grounding.
Answer: The speed is not mentioned in the context. Given the calm seas and autopilot use, the vessel was likely traveling at a typical speed of 5 to 10 knots.
{"result": "Fail", "reason": "Admits the speed is missing but then offers an estimate as an answer."}

## Output

Reply with one JSON object and nothing else:
{"result": "Pass" or "Fail", "reason": "<one sentence>"}
