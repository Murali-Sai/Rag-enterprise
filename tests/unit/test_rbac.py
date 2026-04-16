from langchain_core.documents import Document

from src.auth.rbac import (
    check_department_access,
    filter_documents_by_access,
    get_accessible_departments,
    get_information_barriers_for_user,
)


def _make_doc(department: str, content: str = "test") -> Document:
    return Document(page_content=content, metadata={"department": department})


class TestGetAccessibleDepartments:
    def test_admin_sees_all(self):
        deps = get_accessible_departments({"admin"})
        assert "sec_filings" in deps
        assert "risk_management" in deps
        assert "compliance" in deps
        assert "research" in deps
        assert "trading" in deps

    def test_trading_role(self):
        deps = get_accessible_departments({"trading"})
        assert "trading" in deps
        assert "risk_management" in deps
        assert "sec_filings" in deps
        assert "compliance" not in deps
        assert "research" not in deps

    def test_research_role_chinese_wall(self):
        """Research analyst should NOT see trading or compliance (Chinese Wall)."""
        deps = get_accessible_departments({"research"})
        assert "research" in deps
        assert "sec_filings" in deps
        assert "trading" not in deps
        assert "compliance" not in deps

    def test_compliance_role(self):
        deps = get_accessible_departments({"compliance"})
        assert "compliance" in deps
        assert "sec_filings" in deps
        assert "risk_management" in deps
        assert "trading" not in deps

    def test_risk_role_broad_access(self):
        deps = get_accessible_departments({"risk"})
        assert "risk_management" in deps
        assert "trading" in deps
        assert "sec_filings" in deps
        assert "compliance" in deps

    def test_viewer_only_public(self):
        deps = get_accessible_departments({"viewer"})
        assert "sec_filings" in deps
        assert "general" in deps
        assert "trading" not in deps
        assert "compliance" not in deps

    def test_wealth_management_limited(self):
        deps = get_accessible_departments({"wealth_management"})
        assert "research" in deps
        assert "sec_filings" in deps
        assert "trading" not in deps

    def test_unknown_role(self):
        deps = get_accessible_departments({"unknown_role"})
        assert deps == set()


class TestInformationBarriers:
    def test_research_trading_wall(self):
        """Research + trading combo should still block trading access (Chinese Wall absolute)."""
        deps = get_accessible_departments({"research", "trading"})
        # The information barrier should remove trading for the research role
        # but trading role grants it back — barrier only blocks the research role
        # Actually, barriers check if blocked_role is in user_roles
        # So if research is in roles, trading is removed regardless
        assert "trading" not in deps

    def test_admin_bypasses_barriers(self):
        deps = get_accessible_departments({"admin", "research"})
        assert "trading" in deps  # Admin overrides barriers

    def test_get_barriers_for_research(self):
        barriers = get_information_barriers_for_user({"research"})
        assert len(barriers) == 2
        names = [b["name"] for b in barriers]
        assert "Research-Trading Wall" in names
        assert "Research-Compliance Wall" in names

    def test_no_barriers_for_trading(self):
        barriers = get_information_barriers_for_user({"trading"})
        assert len(barriers) == 0

    def test_no_barriers_for_admin(self):
        barriers = get_information_barriers_for_user({"admin"})
        assert len(barriers) == 0


class TestFilterDocumentsByAccess:
    def test_admin_sees_all_docs(self):
        docs = [_make_doc("trading"), _make_doc("research"), _make_doc("compliance")]
        filtered = filter_documents_by_access(docs, {"admin"})
        assert len(filtered) == 3

    def test_research_cannot_see_trading(self):
        docs = [_make_doc("trading"), _make_doc("research"), _make_doc("sec_filings")]
        filtered = filter_documents_by_access(docs, {"research"})
        depts = [d.metadata["department"] for d in filtered]
        assert "trading" not in depts
        assert "research" in depts
        assert "sec_filings" in depts

    def test_compliance_cannot_see_trading(self):
        docs = [_make_doc("trading"), _make_doc("compliance"), _make_doc("sec_filings")]
        filtered = filter_documents_by_access(docs, {"compliance"})
        depts = [d.metadata["department"] for d in filtered]
        assert "trading" not in depts
        assert "compliance" in depts

    def test_trader_sees_risk_and_trading(self):
        docs = [_make_doc("trading"), _make_doc("risk_management"), _make_doc("research")]
        filtered = filter_documents_by_access(docs, {"trading"})
        depts = [d.metadata["department"] for d in filtered]
        assert "trading" in depts
        assert "risk_management" in depts
        assert "research" not in depts

    def test_empty_docs(self):
        filtered = filter_documents_by_access([], {"admin"})
        assert filtered == []


class TestCheckDepartmentAccess:
    def test_admin_has_all_access(self):
        assert check_department_access({"admin"}, "trading") is True
        assert check_department_access({"admin"}, "compliance") is True

    def test_trader_has_trading_access(self):
        assert check_department_access({"trading"}, "trading") is True

    def test_research_no_trading_access(self):
        assert check_department_access({"research"}, "trading") is False

    def test_viewer_no_trading_access(self):
        assert check_department_access({"viewer"}, "trading") is False
        assert check_department_access({"viewer"}, "compliance") is False

    def test_viewer_has_sec_filings(self):
        assert check_department_access({"viewer"}, "sec_filings") is True
