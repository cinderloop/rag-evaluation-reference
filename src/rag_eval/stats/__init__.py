"""Statistics for evaluation results, standard library only (see ADR 0011).

- ``intervals``: Wilson and percentile bootstrap intervals for pass rates and means.
- ``clustered``: standard errors and bootstrap intervals that respect repeated runs of
  the same question.
- ``paired``: comparing two configs on the same items (exact McNemar, paired bootstrap).
- ``consistency``: agreement of repeated runs.
- ``power``: minimum detectable effect for a given number of items.
"""
