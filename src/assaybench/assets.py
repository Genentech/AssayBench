"""Third-party data assets that are kept outside the package.

Some of the data AssayBench evaluates against cannot ship inside the wheel.
MSigDB is the clearest case: its C2 canonical-pathways collection mixes CC BY
4.0 sets with KEGG sets under Kanehisa Laboratories' own terms, KEGG MEDICUS
sets under a share-alike licence, and BioCarta sets held by qualified
permission. Bundling the file would mean redistributing all of that under this
package's MIT licence, which we are not entitled to do. Assets remain subject
to their providers' terms.

So: downloadable assets are fetched into a user-level cache. Assets without a
provider-operated machine-download URL must be downloaded manually from the
provider and placed at the path named by AssayBench.

    $ assaybench download msigdb-go-bp

Everything here fails loudly. :func:`asset_path` raises
:class:`MissingAsset` when an asset has not been downloaded; it never
substitutes a default, degrades to a stand-in, or returns ``None``. A metric
computed against silently-swapped reference data is worse than no metric,
because it still produces a plausible number. Callers that genuinely want an
optional dependency must catch :class:`MissingAsset` themselves and say so in
their output.

Downloads are checksum-verified against the exact revisions used for the
published results, so a provider re-cutting a release is detected rather than
quietly changing your numbers.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Asset",
    "MissingAsset",
    "ChecksumMismatch",
    "ASSETS",
    "cache_dir",
    "asset_path",
    "download",
    "ensure_asset",
    "is_downloaded",
    "iter_assets",
]


class MissingAsset(FileNotFoundError):
    """An asset is needed but has not been made available locally.

    Carries instructions for obtaining and placing the file.
    """


class ChecksumMismatch(RuntimeError):
    """An asset file did not match its recorded sha256.

    Either the provider re-cut the release under the same URL, or the transfer
    was corrupted. Both mean the file must not be used: results computed
    against it would not be the published ones.
    """


@dataclass(frozen=True)
class Asset:
    """A third-party file kept outside the Python distribution.

    Attributes:
        name: Registry key, and what the user passes to ``assaybench download``.
        filename: Name the file is stored under inside the cache.
        url: Machine-download URL, or ``None`` when the provider requires a
            manual download.
        sha256: Checksum of the exact revision used for the published results.
        size: Expected size in bytes, for the progress message.
        summary: One line on what the asset is for.
        license: The provider's terms, stated so a user knows what they are
            accepting by running the download.
        homepage: Where those terms are published.
        manual_download_page: Provider page where a user can obtain an asset
            that has no machine-download URL.
        path_env: Optional environment variable pointing directly at a
            manually supplied file.
    """

    name: str
    filename: str
    url: str | None
    sha256: str
    size: int
    summary: str
    license: str
    homepage: str
    manual_download_page: str | None = None
    path_env: str | None = None


# MSigDB 2023.2.Hs -- the release used for the AssayLoop pathway-overlap
# results. The Broad serves these without registration, but the collections
# differ sharply in what they permit downstream, which is why the licence is
# recorded per asset rather than once for the set.
_MSIGDB = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs"

ASSETS: dict[str, Asset] = {
    asset.name: asset
    for asset in (
        Asset(
            name="depmap-common-essentials-26q1",
            filename="CRISPRInferredCommonEssentials.csv",
            url=None,
            sha256="c21c92c52579182a7c25bcab1f7e9ec0a54cd09d10677f7e0f1e42558303e6c8",
            size=25_021,
            summary=(
                "DepMap Public 26Q1 common-essential genes used by the "
                "%essential metric. Must be downloaded from DepMap manually."
            ),
            license=(
                "DepMap Terms of Use: by downloading or using this asset you "
                "must comply with those terms; clinical and commercial use "
                "is restricted. Not covered by AssayBench's MIT licence."
            ),
            homepage="https://depmap.org/portal/terms/",
            manual_download_page="https://depmap.org/portal/download/",
            path_env="ASSAYBENCH_DEPMAP_PATH",
        ),
        Asset(
            name="msigdb-go-bp",
            filename="c5.go.bp.v2023.2.Hs.symbols.gmt",
            url=f"{_MSIGDB}/c5.go.bp.v2023.2.Hs.symbols.gmt",
            sha256="b4b3b49a267c2239fad40d39a56f2146a57743992943915e80ebed34ec7ebaae",
            size=4_929_531,
            summary=(
                "GO biological process, 7,647 sets. Broadest gene coverage, and "
                "the default source for pathway overlap."
            ),
            license="CC BY 4.0",
            homepage="https://www.gsea-msigdb.org/gsea/msigdb",
        ),
        Asset(
            name="msigdb-hallmark",
            filename="h.all.v2023.2.Hs.symbols.gmt",
            url=f"{_MSIGDB}/h.all.v2023.2.Hs.symbols.gmt",
            sha256="8ff1f036f0988c99d42cf9e5ea24d355f10b8ece5e0d4f6edf26a0458f480780",
            size=48_690,
            summary="Hallmark, 50 coarse well-curated sets.",
            license="CC BY 4.0",
            homepage="https://www.gsea-msigdb.org/gsea/msigdb",
        ),
        Asset(
            name="msigdb-canonical-pathways",
            filename="c2.cp.v2023.2.Hs.symbols.gmt",
            url=f"{_MSIGDB}/c2.cp.v2023.2.Hs.symbols.gmt",
            sha256="4826118ed2df8af485d524395425a77d5563681417097bc38c3393640a5bde54",
            size=1_541_220,
            summary=(
                "Curated canonical pathways, 3,795 sets: Reactome, KEGG, "
                "WikiPathways, BioCarta, PID."
            ),
            license=(
                "Mixed, and not all of it is free to redistribute. 1,692 "
                "Reactome and 791 WikiPathways sets are CC BY 4.0; 619 "
                "KEGG_MEDICUS sets are CC BY-SA 4.0; 186 legacy KEGG sets are "
                "subject to Kanehisa Laboratories' academic and commercial "
                "terms; 292 BioCarta sets are held by qualified permission. "
                "Check your own position before republishing anything derived "
                "from this file."
            ),
            homepage="https://www.gsea-msigdb.org/gsea/msigdb",
        ),
    )
}


def cache_dir() -> Path:
    """Directory cached assets live in.

    ``$ASSAYBENCH_CACHE`` if set, else ``$XDG_CACHE_HOME/assaybench``, else
    ``~/.cache/assaybench``. The directory is not created here.
    """
    env = os.environ.get("ASSAYBENCH_CACHE")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "assaybench"


def iter_assets() -> Iterator[Asset]:
    """Every registered asset, in registration order."""
    return iter(ASSETS.values())


def _get(name: str) -> Asset:
    try:
        return ASSETS[name]
    except KeyError:
        known = ", ".join(ASSETS)
        raise KeyError(
            f"Unknown asset {name!r}. Registered assets: {known}."
        ) from None


def is_downloaded(name: str) -> bool:
    """Whether ``name`` is present in the cache. Does not verify the checksum."""
    return _configured_path(_get(name)).is_file()


def _configured_path(asset: Asset) -> Path:
    """The configured local path for an asset."""
    if asset.path_env:
        configured = os.environ.get(asset.path_env)
        if configured:
            return Path(configured).expanduser()
    return cache_dir() / asset.filename


def _missing_message(asset: Asset, path: Path) -> str:
    if asset.url is None:
        source = asset.manual_download_page or asset.homepage
        placement = (
            f"  Or set:      ${asset.path_env}=/path/to/{asset.filename}\n"
            if asset.path_env
            else ""
        )
        action = (
            f"  Download:    {source}\n"
            f"  Save as:     {path}\n"
            f"{placement}"
        )
    else:
        action = f"  Fetch it:    assaybench download {asset.name}\n"
    return (
        f"Asset {asset.name!r} ({asset.summary.split('.')[0]}) is not available.\n"
        f"  Expected at: {path}\n"
        f"{action}"
        f"  Terms:       {asset.homepage}\n"
        f"  Licence:     {asset.license}\n"
        f"Set $ASSAYBENCH_CACHE to use a different cache directory."
    )


def asset_path(name: str) -> Path:
    """Path to an asset available on the local machine.

    Args:
        name: A key of :data:`ASSETS`.

    Returns:
        The path to the cached file.

    Raises:
        KeyError: If ``name`` is not registered.
        MissingAsset: If the asset is not available locally. This is deliberate
            and is never softened into a fallback -- see the module docstring.
    """
    asset = _get(name)
    path = _configured_path(asset)
    if path.is_file():
        return path
    raise MissingAsset(_missing_message(asset, path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    names: str | Iterable[str],
    *,
    force: bool = False,
    quiet: bool = False,
) -> list[Path]:
    """Download assets into the cache, verifying their checksums.

    The download goes to a temporary file in the cache directory and is moved
    into place only after the checksum matches, so an interrupted or corrupted
    transfer never leaves a file that :func:`asset_path` would hand out.

    Args:
        names: One asset name, or an iterable of them.
        force: Re-download even if the file is already present and valid.
        quiet: Suppress progress output.

    Returns:
        The cached paths, in the order requested.

    Raises:
        KeyError: If a name is not registered.
        MissingAsset: If an explicitly requested asset requires manual
            acquisition from its provider.
        ChecksumMismatch: If a downloaded file does not match its sha256. The
            bad file is removed.
    """
    if isinstance(names, str):
        names = [names]

    root = cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    paths = []

    for name in names:
        asset = _get(name)
        if asset.url is None:
            raise MissingAsset(_missing_message(asset, _configured_path(asset)))
        dest = root / asset.filename
        url = asset.url

        if dest.is_file() and not force:
            if _sha256(dest) == asset.sha256:
                if not quiet:
                    print(f"  [have] {name} -> {dest}")
                paths.append(dest)
                continue
            if not quiet:
                print(f"  [bad ] {name} checksum mismatch, re-downloading")

        if not quiet:
            print(f"  [get ] {name} ({asset.size / 1e6:.1f} MB) from {url}")
            print(f"         terms: {asset.license}")
            print(f"                {asset.homepage}")

        fd, tmp_name = tempfile.mkstemp(dir=root, prefix=f".{asset.filename}.")
        tmp = Path(tmp_name)
        os.close(fd)
        try:
            with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
                shutil.copyfileobj(response, out)
            actual = _sha256(tmp)
            if actual != asset.sha256:
                raise ChecksumMismatch(
                    f"{name}: downloaded file does not match the recorded "
                    f"revision.\n"
                    f"  url:      {url}\n"
                    f"  expected: {asset.sha256}\n"
                    f"  actual:   {actual}\n"
                    f"The provider may have re-cut this release. Results "
                    f"computed against it would not be the published ones."
                )
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)

        if not quiet:
            print(f"         -> {dest}")
        paths.append(dest)

    return paths


def ensure_asset(name: str, *, quiet: bool = False) -> Path:
    """Return a verified local asset, downloading only when permitted.

    Assets with a machine-download URL are fetched when missing. Assets marked
    for manual acquisition must already be at their configured path. Every
    returned file is checksum-verified.
    """
    asset = _get(name)
    if asset.url is not None:
        return download(name, quiet=quiet)[0]

    path = asset_path(name)
    actual = _sha256(path)
    if actual != asset.sha256:
        raise ChecksumMismatch(
            f"{name}: manually supplied file does not match the recorded "
            f"revision.\n"
            f"  path:     {path}\n"
            f"  expected: {asset.sha256}\n"
            f"  actual:   {actual}\n"
            f"Download the exact release named in the asset's provenance."
        )
    if not quiet:
        print(f"  [have] {name} -> {path}")
    return path
