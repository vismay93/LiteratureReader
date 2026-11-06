import hashlib
import io
import json
import tarfile

import numpy as np
import pytest
import soundfile as sf

import config
from pipeline import bundle

PDF = b"%PDF-1.4 fake paper"
PID = hashlib.sha256(PDF).hexdigest()[:16]


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "lib")
    d = config.DATA_DIR / PID
    for sub in ("audio", "crops", "narration", "layout"):
        (d / sub).mkdir(parents=True)
    (d / "source.pdf").write_bytes(PDF)
    sf.write(d / "audio" / "p0b0.wav", np.zeros(2400, dtype=np.float32), 24000)
    (d / "audio" / "p0b0.json").write_text('{"duration": 0.1, "sentences": []}')
    (d / "layout" / "0.json").write_text("[]")
    doc = {"id": PID, "title": "T", "pages": [{"width": 1, "height": 1}],
           "blocks": [{"id": "p0b0", "page": 0, "type": "text", "bbox": [0, 0, 1, 1],
                       "text": "hi", "audio": "audio/p0b0.wav", "duration": 0.1}]}
    (d / "doc.json").write_text(json.dumps(doc))
    return tmp_path


def _tar(path, members):
    with tarfile.open(path, "w:gz") as t:
        for name, data in members:
            if isinstance(data, tarfile.TarInfo):
                t.addfile(data)
                continue
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    return path


@pytest.mark.parametrize("audio", ["wav", "mp3"])
def test_roundtrip(lib, monkeypatch, audio):
    out = bundle.export_paper(PID, lib / "exp", audio=audio)
    assert bundle.export_paper(PID, lib / "exp", audio=audio) is None  # up to date
    monkeypatch.setattr(config, "DATA_DIR", lib / "other")
    assert bundle.import_bundle(out) == (PID, "imported")
    doc = json.loads((lib / "other" / PID / "doc.json").read_text())
    assert doc["blocks"][0]["audio"] == f"audio/p0b0.{audio}"
    assert (lib / "other" / PID / doc["blocks"][0]["audio"]).exists()
    assert bundle.import_bundle(out) == (PID, "skipped")
    assert bundle.import_bundle(out, force=True) == (PID, "imported")


def _doc():
    return json.dumps({"id": PID, "blocks": []}).encode()


@pytest.mark.parametrize("bad", [
    f"{PID}/../../evil.txt",
    f"/{PID}/doc.json",
    f"{PID}/audio/../../x.wav",
    f"{PID}/extra.sh",
    "0123456789abcdef/doc.json",  # second paper id
])
def test_rejects_bad_paths(lib, bad):
    f = _tar(lib / "b.tar.gz", [(f"{PID}/doc.json", _doc()), (f"{PID}/source.pdf", PDF), (bad, b"x")])
    with pytest.raises(bundle.BundleError):
        bundle.import_bundle(f)
    assert not (lib / "evil.txt").exists()


def test_rejects_symlink(lib):
    link = tarfile.TarInfo(f"{PID}/audio/p0b0.wav")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    f = _tar(lib / "b.tar.gz", [(f"{PID}/doc.json", _doc()), (f"{PID}/source.pdf", PDF), ("", link)])
    with pytest.raises(bundle.BundleError):
        bundle.import_bundle(f)


def test_rejects_hash_mismatch(lib, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", lib / "other")
    f = _tar(lib / "b.tar.gz", [(f"{PID}/doc.json", _doc()), (f"{PID}/source.pdf", b"%PDF other")])
    with pytest.raises(bundle.BundleError, match="does not match"):
        bundle.import_bundle(f)
    assert not (lib / "other" / PID).exists()


def test_rejects_missing_audio(lib, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", lib / "other")
    doc = json.dumps({"id": PID, "blocks": [{"id": "p0b0", "audio": "audio/p0b0.wav"}]}).encode()
    f = _tar(lib / "b.tar.gz", [(f"{PID}/doc.json", doc), (f"{PID}/source.pdf", PDF)])
    with pytest.raises(bundle.BundleError, match="missing"):
        bundle.import_bundle(f)
