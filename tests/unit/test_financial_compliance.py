from src.guardrails.financial_compliance import (
    apply_financial_disclaimers,
    check_financial_compliance,
)


class TestFinancialCompliance:
    def test_clean_response(self):
        result = check_financial_compliance(
            query="What is the CET1 ratio?",
            response="The CET1 ratio was 13.2% as of year-end.",
            user_roles={"risk"},
        )
        assert not result.requires_investment_disclaimer
        assert not result.requires_mnpi_warning
        assert len(result.flags) == 0

    def test_investment_advice_detected(self):
        result = check_financial_compliance(
            query="Should I buy TECH?",
            response="You should buy TECH stock because the price target is $285.",
            user_roles={"research"},
        )
        assert result.requires_investment_disclaimer
        assert "investment_advice_detected" in result.flags

    def test_guaranteed_return_blocked(self):
        result = check_financial_compliance(
            query="What returns can I expect?",
            response="This is a guaranteed return of 15% annually.",
            user_roles={"wealth_management"},
        )
        assert result.contains_prohibited_advice
        assert "prohibited_guarantee_language" in result.flags

    def test_mnpi_detected(self):
        result = check_financial_compliance(
            query="What about the upcoming merger?",
            response="The upcoming merger hasn't been announced yet but sources indicate...",
            user_roles={"trading"},
        )
        assert result.requires_mnpi_warning
        assert "potential_mnpi_detected" in result.flags

    def test_forward_looking_statement(self):
        result = check_financial_compliance(
            query="What is the revenue outlook?",
            response="We expect revenue to grow 15% next year driven by strong demand.",
            user_roles={"research"},
        )
        assert result.requires_forward_looking_disclaimer
        assert "forward_looking_statement" in result.flags

    def test_apply_disclaimers_normal(self):
        result = check_financial_compliance(
            query="What is the outlook?",
            response="Revenue is projected to increase next quarter.",
            user_roles={"research"},
        )
        output = apply_financial_disclaimers("Revenue is projected to increase.", result)
        assert "forward-looking statements" in output

    def test_apply_disclaimers_prohibited(self):
        result = check_financial_compliance(
            query="Give me guaranteed returns",
            response="This is a guaranteed return investment.",
            user_roles={"wealth_management"},
        )
        output = apply_financial_disclaimers("Guaranteed return of 20%.", result)
        assert "cannot provide guaranteed" in output
        assert "does not constitute investment advice" in output
