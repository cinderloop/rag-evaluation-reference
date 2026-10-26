"""Binary LLM judges. DeepEval telemetry is turned off before DeepEval is imported."""

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
