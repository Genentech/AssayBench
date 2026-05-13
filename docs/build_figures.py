"""Convert the paper's figure PDFs to web-friendly PNGs.

Run this once after dropping the arXiv tarball ``figures/`` directory
somewhere local. By default it expects the figures next to this repo, but
``--source-dir`` lets you point at any directory.

Requires PyMuPDF (``pip install pymupdf``).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing Figure*.pdf and *.png from the arXiv source.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "assets" / "figures",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    import fitz  # type: ignore  # noqa: PLC0415

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(args.source_dir.glob("*.pdf")):
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=args.dpi)
        target = out_dir / f"{pdf_path.stem}.png"
        pix.save(str(target))
        print(f"{pdf_path.name} -> {target} ({pix.width}x{pix.height})")
        doc.close()

    for png_path in sorted(args.source_dir.glob("*.png")):
        target = out_dir / png_path.name
        shutil.copy(png_path, target)
        print(f"copied {png_path.name} -> {target}")


if __name__ == "__main__":
    main()
