"""Is this question about a company it never names?

"How much did the bank set aside for credit losses?" has two answers in this
corpus and the question contains nothing to choose between them. The measured
behaviour before this module existed: the pipeline retrieved whichever bank's
chunks scored better, the model answered fluently about that one, and nothing
anywhere flagged that a coin had been tossed. The ambiguous stratum held at
0.667 refusal correctness across every baseline ever recorded — the only
answerable stratum no retrieval fix moved — because this is not a retrieval
failure. Retrieval was asked an underspecified question and answered it.

Detection is mechanical, same argument as `entities.py`: an LLM classifier in
front of retrieval would put another model's judgment inside the measurement
chain and need re-measuring every time that model changed. Two literal
signals, both gated on `detect_entities()` finding nothing — a named company
always wins:

- **A definite reference to an unnamed company.** "The company", "the bank":
  grammar that points at an antecedent the question never supplies. Five
  filings match "the company" and two match "the bank".
- **A company-scoped financial metric with no subject at all.** "What was
  total revenue last year?" names neither a company nor anything else — the
  capitalized-token check is what keeps this branch away from "What was
  Netflix's streaming revenue?", which names a company this corpus simply
  does not hold. That question is out-of-corpus, not ambiguous, and its
  refusal already works: the model declined 24 of 24 such probes unaided
  (scripts/probe_refusal.py). This module must not reroute a failure mode
  that is measured and handled into one that guesses at wording.

Both heuristics are deliberately biased toward not firing. A false negative
falls through to today's behaviour — an arbitrary but fluent answer — while a
false positive refuses a question the corpus answers, which is the failure
`_refusal_correctness` exists to catch on the answerable side. When in doubt,
this module stays silent.
"""

import re

from src.edgar.client import COMPANY_REGISTRY
from src.retrieval.entities import detect_entities

# Grammar that points at a specific company without naming one. Possessives
# match too: `\b` sits between "company" and the apostrophe in "the company's".
_DEFINITE_REFERENCE = re.compile(
    r"\bthe\s+(?:company|companies|firm|firms|bank|banks)\b",
    re.IGNORECASE,
)

# Metrics every filing in the corpus reports. A question asking for one of
# these with no subject admits one answer per company. Kept short on purpose:
# each term is a chance to refuse an answerable question, and the suite only
# demands "revenue". Terms like "rate" or "limits" are excluded because they
# appear in questions this corpus answers without naming any company at all —
# the internal policy documents.
_FINANCIAL_METRIC = re.compile(
    r"\b(?:revenue|revenues|net income|profit|profits|margin|margins|earnings|"
    r"credit losses)\b",
    re.IGNORECASE,
)


def _names_anything(query: str) -> bool:
    """A capitalized token after the first word — some proper noun is present.

    Cruder than NER and meant to be. "Netflix", "FOMC", "ACME Financial
    Holdings", "GAAP", even a stray "I" all count, and every one of them
    correctly suppresses the no-subject branch: a question that names
    *anything* is not the bare "what was total revenue?" shape this branch
    exists for. Over-counting here only ever produces silence, which is the
    cheap failure.
    """
    words = query.split()
    return any(word[0].isupper() for word in words[1:] if word and word[0].isalpha())


def is_underspecified(query: str) -> bool:
    """True when the question needs a company and supplies none."""
    if detect_entities(query):
        return False
    if _DEFINITE_REFERENCE.search(query):
        return True
    return bool(_FINANCIAL_METRIC.search(query)) and not _names_anything(query)


def clarification_detail() -> str:
    """The one thing the asker can do about it, with the choices spelled out.

    Built from the registry rather than written out so that adding a sixth
    company cannot leave this message listing five.
    """
    names = ", ".join(company["name"] for company in COMPANY_REGISTRY.values())
    return f"The corpus holds 10-K filings for: {names}. Naming one will get an answer."
