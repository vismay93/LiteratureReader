import pytest
from batch import expand_inputs


def test_expand_inputs(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "b.pdf").write_bytes(b"%PDF")
    lst = tmp_path / "list.txt"
    lst.write_text("# reading list\n1706.03762\nhttps://arxiv.org/abs/2401.00001  # comment\n\n")
    got = expand_inputs([str(tmp_path), f"@{lst}", "1706.03762"])
    assert got == [("pdf", str(tmp_path / "a.pdf")), ("pdf", str(tmp_path / "b.pdf")),
                   ("arxiv", "1706.03762"), ("arxiv", "2401.00001")]


def test_expand_rejects_garbage():
    with pytest.raises(SystemExit):
        expand_inputs(["https://evil.com/x.pdf"])
