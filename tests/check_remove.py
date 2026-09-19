"""remove_model on a fake cache: shared blobs survive while another repo uses them,
unshared ones go, and nothing outside the cache is touched.

Run with: PYTHONPATH=. uv run python tests/check_remove.py
"""
import tempfile
from pathlib import Path

from app import extract, server

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    cache, vecs, outside = tmp / ".cache", tmp / "vectors", tmp / "outside.bin"
    server.CACHE, extract.VECTOR_DIR = cache, vecs
    outside.write_bytes(b"x" * 10)

    def blob(name, size):
        p = cache / "blobs" / name[:2] / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"0" * size)
        return p

    shared, only_a = blob("aa11", 100), blob("bb22", 200)
    for repo, links in {"models--org--a": [shared, only_a, outside], "models--org--b": [shared]}.items():
        snap = cache / repo / "snapshots" / "rev"
        snap.mkdir(parents=True)
        for i, target in enumerate(links):
            (snap / f"f{i}").symlink_to(target)
        (cache / repo / "config.json").write_text("{}")
    vecs.mkdir()
    extract.vectors_path("org/a").write_bytes(b"v")

    assert server.disk_bytes("org/a") == 302, server.disk_bytes("org/a")  # outside file not counted
    server.remove_model("org/a")
    assert not (cache / "models--org--a").exists()
    assert shared.exists(), "blob still used by org/b was deleted"
    assert not only_a.exists(), "unshared blob was left behind"
    assert outside.exists(), "file outside the cache was touched"
    assert not extract.vectors_path("org/a").exists()
    assert server.disk_bytes("org/b") == 102
    print("ok  remove_model frees unshared blobs, keeps shared ones, stays inside the cache")
