import pytest
from pipeline.ingest import IngestError, parse_arxiv, paper_dir, store_pdf


@pytest.mark.parametrize("ref,expected", [
    ("1706.03762", "1706.03762"),
    ("arXiv:1706.03762v7", "1706.03762v7"),
    ("https://arxiv.org/abs/1706.03762", "1706.03762"),
    ("https://arxiv.org/pdf/1706.03762v2.pdf", "1706.03762v2"),
    ("arxiv.org/pdf/2401.00001", "2401.00001"),
    ("https://arxiv.org/abs/cs.LG/0601001", "cs.LG/0601001"),
])
def test_parse_arxiv_ok(ref, expected):
    assert parse_arxiv(ref) == expected


@pytest.mark.parametrize("ref", [
    "https://evil.com/abs/1706.03762",
    "https://arxiv.org.evil.com/abs/1706.03762",
    "file:///etc/passwd",
    "http://127.0.0.1:8081/abs/1706.03762",
    "https://arxiv.org/abs/../../etc",
])
def test_parse_arxiv_rejects(ref):
    with pytest.raises(IngestError):
        parse_arxiv(ref)


def test_store_rejects_non_pdf(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with pytest.raises(IngestError):
        store_pdf(b"hello")
    pid = store_pdf(b"%PDF-1.4 test")
    assert (tmp_path / pid / "source.pdf").exists()


def test_paper_dir_rejects_traversal():
    with pytest.raises(IngestError):
        paper_dir("../../etc")
