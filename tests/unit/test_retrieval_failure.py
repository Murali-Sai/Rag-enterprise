"""The serving path's behaviour when retrieval itself cannot run.

Written after a live incident: the OpenAI balance reached zero, every query on
the public demo returned a bare "Internal Server Error", and nothing in the
response said whether the question, the index, or the deployment was at fault.
Embedding a question costs $0.000005, which is not a reason to leave the error
path around it unwritten.

The properties worth pinning are that the request still returns a well-formed
response, that the response says which of the three things went wrong, and
that the audit trail records it — a failed query is still a query, and 17a-4
does not have an exception for the ones that errored.
"""

import pytest

from src.api.routes.query import _retrieval_failure


class TestClassification:
    @pytest.mark.parametrize(
        "message",
        [
            "RateLimitError: Error code: 429 - You have no credits remaining.",
            "Error code: 429 - {'error': {'code': 'insufficient_quota'}}",
            "openai.RateLimitError: credit_balance_exhausted",
            "ResourceExhausted: 429 You exceeded your current quota",
        ],
    )
    def test_a_billing_failure_says_so_without_blaming_the_question(self, message):
        reason, flag = _retrieval_failure(RuntimeError(message))

        assert "quota" in flag
        assert "temporarily unavailable" in reason.lower()
        # The reassurance is the point: a reader must not conclude the corpus
        # or the published numbers are affected.
        assert "unaffected" in reason.lower()

    @pytest.mark.parametrize(
        "message",
        [
            "AuthenticationError: Incorrect API key provided",
            "Error code: 401 - invalid_api_key",
        ],
    )
    def test_a_credentials_failure_is_named_as_a_server_misconfiguration(self, message):
        reason, flag = _retrieval_failure(RuntimeError(message))

        assert "auth" in flag
        assert "credentials" in reason.lower()
        assert "not a problem with your question" in reason.lower()

    def test_an_unrecognised_failure_is_reported_as_a_server_error(self):
        reason, flag = _retrieval_failure(ValueError("collection dimension mismatch"))

        assert "ValueError" in flag
        assert "server-side error" in reason.lower()

    def test_every_case_carries_the_exception_type_into_the_flag_or_a_named_cause(self):
        """The audit trail has to be able to tell these apart after the fact."""
        flags = {
            _retrieval_failure(RuntimeError("insufficient_quota"))[1],
            _retrieval_failure(RuntimeError("invalid_api_key"))[1],
            _retrieval_failure(ValueError("something else"))[1],
        }

        assert len(flags) == 3
        assert all(f.startswith("retrieval_unavailable") for f in flags)
