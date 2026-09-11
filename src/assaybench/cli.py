"""``assaybench`` command line entry point.

Currently this exists for one job: managing third-party data assets that the
package is not entitled to redistribute. See :mod:`assaybench.assets`.

    $ assaybench assets                        # what exists, what is cached
    $ assaybench download msigdb-go-bp         # fetch one
    $ assaybench download --all                # fetch everything
"""

from __future__ import annotations

import argparse
import sys

from assaybench import assets


def _cmd_assets(args: argparse.Namespace) -> int:
    root = assets.cache_dir()
    print(f"cache: {root}\n")
    for asset in assets.iter_assets():
        state = "cached " if assets.is_downloaded(asset.name) else "missing"
        print(f"[{state}] {asset.name}  ({asset.size / 1e6:.1f} MB)")
        print(f"          {asset.summary}")
        print(f"          licence: {asset.license}")
        if asset.url is None and asset.manual_download_page:
            print(f"          manual download: {asset.manual_download_page}")
        print()
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    if args.all:
        names = [asset.name for asset in assets.iter_assets() if asset.url is not None]
    elif args.names:
        names = args.names
    else:
        print("error: name an asset, or pass --all. `assaybench assets` lists them.",
              file=sys.stderr)
        return 2

    try:
        assets.download(names, force=args.force)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    except assets.MissingAsset as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except assets.ChecksumMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assaybench",
        description="AssayBench command line tools.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_assets = sub.add_parser(
        "assets", help="list third-party data assets and whether they are cached"
    )
    p_assets.set_defaults(func=_cmd_assets)

    p_download = sub.add_parser(
        "download",
        help="download third-party data assets into the local cache",
        description=(
            "Fetch automatically downloadable data that AssayBench does not "
            "redistribute. Some registered assets require manual acquisition; "
            "`assaybench assets` identifies them and prints their terms."
        ),
    )
    p_download.add_argument("names", nargs="*", help="asset names to download")
    p_download.add_argument(
        "--all", action="store_true", help="download every automatic asset"
    )
    p_download.add_argument(
        "--force", action="store_true", help="re-download even if already cached"
    )
    p_download.set_defaults(func=_cmd_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
