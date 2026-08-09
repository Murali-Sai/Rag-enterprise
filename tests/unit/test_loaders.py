"""Document loading, and the encoding it reads with.

`TextLoader` and `BSHTMLLoader` default to `locale.getpreferredencoding()`,
which is cp1252 on Windows and UTF-8 on Linux. The same file therefore
ingested differently on a dev machine and in the Docker build, and the index
currently on disk carries the evidence: the sample 10-K's em dashes are
stored as "â€"". Nothing failed — mojibake is a successful read of the wrong
bytes — so this is pinned by a test rather than left to be noticed again.
"""

import pytest

from src.common.exceptions import DocumentIngestionError
from src.ingestion.loaders import get_supported_extensions, load_document

# U+2014 EM DASH: e2 80 94 in UTF-8. Read as cp1252 it becomes "â€"", which
# is exactly the corruption found in the live index.
EM_DASH_LINE = "ACME FINANCIAL HOLDINGS, INC.\nFORM 10-K — ANNUAL REPORT\n"


@pytest.fixture
def utf8_file(tmp_path):
    path = tmp_path / "annual_report_10k.txt"
    path.write_text(EM_DASH_LINE, encoding="utf-8")
    return path


class TestEncoding:
    def test_utf8_survives_a_txt_load(self, utf8_file):
        (document,) = load_document(utf8_file)

        assert "—" in document.page_content
        assert "â€" not in document.page_content

    def test_utf8_survives_an_html_load(self, tmp_path):
        path = tmp_path / "filing.html"
        path.write_text(f"<html><body><p>{EM_DASH_LINE}</p></body></html>", encoding="utf-8")

        documents = load_document(path)

        assert "—" in documents[0].page_content
        assert "â€" not in documents[0].page_content

    def test_it_does_not_depend_on_the_platform_locale(self, utf8_file, monkeypatch):
        """The failure mode is environmental: correct on the machine that
        wrote the file, corrupt on the one that ingests it."""
        monkeypatch.setattr("locale.getpreferredencoding", lambda *_: "cp1252")

        (document,) = load_document(utf8_file)

        assert "—" in document.page_content


class TestRegistry:
    def test_an_unsupported_type_names_what_is_supported(self, tmp_path):
        path = tmp_path / "filing.xlsx"
        path.write_text("x", encoding="utf-8")

        with pytest.raises(DocumentIngestionError, match="Unsupported file type"):
            load_document(path)

    def test_a_missing_file_is_not_an_unsupported_type(self, tmp_path):
        with pytest.raises(DocumentIngestionError, match="File not found"):
            load_document(tmp_path / "absent.txt")

    def test_the_supported_set_is_reported(self):
        assert set(get_supported_extensions()) >= {".pdf", ".docx", ".txt", ".md", ".html"}
