"""Tests for the downloaded-asset cache.

The behaviour under test is mostly negative: an asset that has not been
downloaded must produce an error naming the fix, and a corrupted download must
never end up in the cache. No test here touches the network.
"""

from __future__ import annotations

import hashlib

import pytest

from assaybench import assets


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSAYBENCH_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("ASSAYBENCH_DEPMAP_PATH", raising=False)
    return tmp_path / "cache"


def test_cache_dir_follows_the_env_var(isolated_cache):
    assert assets.cache_dir() == isolated_cache


def test_cache_dir_falls_back_to_xdg(tmp_path, monkeypatch):
    monkeypatch.delenv("ASSAYBENCH_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert assets.cache_dir() == tmp_path / "assaybench"


def test_missing_asset_raises_rather_than_falling_back():
    with pytest.raises(assets.MissingAsset):
        assets.asset_path("msigdb-go-bp")


def test_missing_asset_error_names_the_fix():
    with pytest.raises(assets.MissingAsset) as excinfo:
        assets.asset_path("msigdb-go-bp")
    message = str(excinfo.value)
    assert "assaybench download msigdb-go-bp" in message
    assert "gsea-msigdb" in message
    assert "CC BY 4.0" in message


def test_unknown_asset_lists_the_registered_ones():
    with pytest.raises(KeyError) as excinfo:
        assets.asset_path("not-an-asset")
    assert "msigdb-go-bp" in excinfo.value.args[0]


def test_is_downloaded_reflects_the_cache(isolated_cache):
    assert not assets.is_downloaded("msigdb-hallmark")
    isolated_cache.mkdir(parents=True)
    (isolated_cache / assets.ASSETS["msigdb-hallmark"].filename).write_text("x")
    assert assets.is_downloaded("msigdb-hallmark")


def test_asset_path_returns_a_cached_file(isolated_cache):
    isolated_cache.mkdir(parents=True)
    target = isolated_cache / assets.ASSETS["msigdb-hallmark"].filename
    target.write_text("x")
    assert assets.asset_path("msigdb-hallmark") == target


def test_registry_checksums_are_well_formed():
    for asset in assets.iter_assets():
        assert len(asset.sha256) == 64
        assert set(asset.sha256) <= set("0123456789abcdef")
        if asset.url is not None:
            assert asset.url.startswith("https://")
        else:
            assert asset.manual_download_page.startswith("https://")
        assert asset.license


def test_c2_licence_flags_the_restricted_collections():
    """The mixed-licence collection must say so; that is the whole reason it is
    downloaded rather than bundled."""
    licence = assets.ASSETS["msigdb-canonical-pathways"].license
    assert "KEGG" in licence
    assert "BioCarta" in licence


def test_depmap_asset_is_pinned_and_not_mit_licensed():
    asset = assets.ASSETS["depmap-common-essentials-26q1"]
    assert asset.sha256 == (
        "c21c92c52579182a7c25bcab1f7e9ec0a54cd09d10677f7e0f1e42558303e6c8"
    )
    assert asset.size == 25_021
    assert asset.url is None
    assert asset.path_env == "ASSAYBENCH_DEPMAP_PATH"
    assert asset.manual_download_page == "https://depmap.org/portal/download/"
    assert "not covered" in asset.license.lower()
    assert asset.homepage == "https://depmap.org/portal/terms/"


# ---------------------------------------------------------------------------
# Download, with the network stubbed out
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self, size: int = -1) -> bytes:
        chunk, self._payload = self._payload, b""
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def serve(monkeypatch):
    """Serve fixed bytes for any URL, and report how many fetches happened."""
    calls = []

    def _serve(payload: bytes):
        def fake_urlopen(url, *args, **kwargs):
            calls.append(url)
            return _FakeResponse(payload)

        monkeypatch.setattr(assets.urllib.request, "urlopen", fake_urlopen)
        return calls

    return _serve


def _pin(monkeypatch, name: str, payload: bytes):
    """Point an asset's recorded checksum at ``payload``."""
    asset = assets.ASSETS[name]
    pinned = type(asset)(
        **{**asset.__dict__, "sha256": hashlib.sha256(payload).hexdigest()}
    )
    monkeypatch.setitem(assets.ASSETS, name, pinned)
    return pinned


def test_download_writes_the_file(monkeypatch, serve, isolated_cache):
    payload = b"SET_A\tdesc\tTP53\tMYC\n"
    _pin(monkeypatch, "msigdb-hallmark", payload)
    serve(payload)

    (path,) = assets.download("msigdb-hallmark", quiet=True)

    assert path.read_bytes() == payload
    assert assets.asset_path("msigdb-hallmark") == path


def test_ensure_asset_downloads_then_uses_verified_cache(
    monkeypatch, serve, isolated_cache
):
    payload = b"SET_A\tdesc\tTP53\n"
    _pin(monkeypatch, "msigdb-hallmark", payload)
    calls = serve(payload)

    first = assets.ensure_asset("msigdb-hallmark", quiet=True)
    second = assets.ensure_asset("msigdb-hallmark", quiet=True)

    assert first == second
    assert first.read_bytes() == payload
    assert len(calls) == 1


def test_manual_asset_can_be_supplied_by_path(monkeypatch, tmp_path):
    payload = b"Essentials\nAAMP (14)\n"
    _pin(monkeypatch, "depmap-common-essentials-26q1", payload)
    source = tmp_path / "CRISPRInferredCommonEssentials.csv"
    source.write_bytes(payload)
    monkeypatch.setenv("ASSAYBENCH_DEPMAP_PATH", str(source))

    path = assets.ensure_asset("depmap-common-essentials-26q1", quiet=True)

    assert path == source


def test_missing_manual_asset_explains_download_and_placement():
    with pytest.raises(assets.MissingAsset) as excinfo:
        assets.ensure_asset("depmap-common-essentials-26q1", quiet=True)

    message = str(excinfo.value)
    assert "https://depmap.org/portal/download/" in message
    assert "CRISPRInferredCommonEssentials.csv" in message
    assert "ASSAYBENCH_DEPMAP_PATH" in message
    assert "https://depmap.org/portal/terms/" in message


def test_manual_asset_is_checksum_verified(monkeypatch, tmp_path):
    source = tmp_path / "CRISPRInferredCommonEssentials.csv"
    source.write_bytes(b"not the pinned release")
    monkeypatch.setenv("ASSAYBENCH_DEPMAP_PATH", str(source))

    with pytest.raises(assets.ChecksumMismatch) as excinfo:
        assets.ensure_asset("depmap-common-essentials-26q1", quiet=True)

    assert "manually supplied file" in str(excinfo.value)
    assert source.is_file()


def test_download_refuses_to_rehost_manual_asset():
    with pytest.raises(assets.MissingAsset) as excinfo:
        assets.download("depmap-common-essentials-26q1", quiet=True)
    assert "Download:" in str(excinfo.value)


def test_download_is_idempotent(monkeypatch, serve, isolated_cache):
    payload = b"SET_A\tdesc\tTP53\n"
    _pin(monkeypatch, "msigdb-hallmark", payload)
    calls = serve(payload)

    assets.download("msigdb-hallmark", quiet=True)
    assets.download("msigdb-hallmark", quiet=True)

    assert len(calls) == 1


def test_download_force_refetches(monkeypatch, serve, isolated_cache):
    payload = b"SET_A\tdesc\tTP53\n"
    _pin(monkeypatch, "msigdb-hallmark", payload)
    calls = serve(payload)

    assets.download("msigdb-hallmark", quiet=True)
    assets.download("msigdb-hallmark", quiet=True, force=True)

    assert len(calls) == 2


def test_checksum_mismatch_raises_and_leaves_no_file(monkeypatch, serve, isolated_cache):
    _pin(monkeypatch, "msigdb-hallmark", b"the right bytes")
    serve(b"the wrong bytes")

    with pytest.raises(assets.ChecksumMismatch) as excinfo:
        assets.download("msigdb-hallmark", quiet=True)

    assert "re-cut this release" in str(excinfo.value)
    assert not assets.is_downloaded("msigdb-hallmark")
    assert list(isolated_cache.iterdir()) == []


def test_corrupt_cached_file_is_replaced(monkeypatch, serve, isolated_cache):
    payload = b"good bytes"
    asset = _pin(monkeypatch, "msigdb-hallmark", payload)
    serve(payload)

    isolated_cache.mkdir(parents=True)
    (isolated_cache / asset.filename).write_bytes(b"truncated")

    (path,) = assets.download("msigdb-hallmark", quiet=True)
    assert path.read_bytes() == payload


def test_download_many(monkeypatch, serve, isolated_cache):
    payload = b"shared"
    _pin(monkeypatch, "msigdb-hallmark", payload)
    _pin(monkeypatch, "msigdb-go-bp", payload)
    serve(payload)

    paths = assets.download(["msigdb-hallmark", "msigdb-go-bp"], quiet=True)
    assert len(paths) == 2
    assert all(p.is_file() for p in paths)


def test_download_unknown_name_raises_before_writing(isolated_cache):
    with pytest.raises(KeyError):
        assets.download("not-an-asset", quiet=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_assets_lists_everything(capsys, isolated_cache):
    from assaybench.cli import main

    assert main(["assets"]) == 0
    out = capsys.readouterr().out
    for asset in assets.iter_assets():
        assert asset.name in out
    assert "missing" in out


def test_cli_download_requires_a_target(capsys, isolated_cache):
    from assaybench.cli import main

    assert main(["download"]) == 2
    assert "--all" in capsys.readouterr().err


def test_cli_download_rejects_unknown_names(capsys, isolated_cache):
    from assaybench.cli import main

    assert main(["download", "not-an-asset"]) == 2
    assert "Unknown asset" in capsys.readouterr().err


def test_cli_download_fetches(monkeypatch, serve, isolated_cache, capsys):
    from assaybench.cli import main

    payload = b"payload"
    _pin(monkeypatch, "msigdb-hallmark", payload)
    serve(payload)

    assert main(["download", "msigdb-hallmark"]) == 0
    assert assets.is_downloaded("msigdb-hallmark")


def test_cli_download_all_skips_manual_assets(monkeypatch, isolated_cache):
    from assaybench.cli import main

    requested = []

    def fake_download(names, **kwargs):
        requested.extend(names)
        return []

    monkeypatch.setattr(assets, "download", fake_download)

    assert main(["download", "--all"]) == 0
    assert "depmap-common-essentials-26q1" not in requested
    assert set(requested) == {
        asset.name for asset in assets.iter_assets() if asset.url is not None
    }


def test_cli_download_explains_manual_asset(capsys, isolated_cache):
    from assaybench.cli import main

    assert main(["download", "depmap-common-essentials-26q1"]) == 2
    err = capsys.readouterr().err
    assert "depmap.org/portal/download" in err
    assert "ASSAYBENCH_DEPMAP_PATH" in err
