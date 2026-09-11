# DepMap common-essential genes

`CRISPRInferredCommonEssentials.csv` — 1,827 gene symbols, used as the set `E`
in the `%essential` metric (`assaybench.benchmark.sequential.percent_essential`).
The file is not distributed, hosted, or downloaded by AssayBench. Each user
must obtain it directly from DepMap and make it available locally.

| | |
|---|---|
| Source | Cancer Dependency Map (DepMap), Broad Institute |
| Portal | https://depmap.org/portal/download/ |
| Release | DepMap Public 26Q1 (2026-04-01) |
| Usage terms | DepMap Terms of Use; not covered by AssayBench's MIT license |
| Asset name | `depmap-common-essentials-26q1` |
| md5 | `f9b12f368abf7684fcd97af31e8a39a2` |
| sha256 | `c21c92c52579182a7c25bcab1f7e9ec0a54cd09d10677f7e0f1e42558303e6c8` |
| Bytes | 25,021 |
| Symbols | 1,827 |

The official DepMap release-file index identifies this exact file as DepMap
Public 26Q1: its published md5 matches the checksum-pinned download. That keeps
the `%essential` numbers byte-reproducible.

## Local setup and terms

Download `CRISPRInferredCommonEssentials.csv` for DepMap Public 26Q1 from the
DepMap portal. Set `$ASSAYBENCH_DEPMAP_PATH` to its local path, or place it in
`$ASSAYBENCH_CACHE`, `$XDG_CACHE_HOME/assaybench`, or
`~/.cache/assaybench`. `load_common_essentials()` checks those locations and
raises with these instructions when the file is missing. Every supplied file
must match the sha256 above or AssayBench refuses to use it.

There is deliberately no URL in AssayBench's asset registry for this file.
Neither the source repository nor a GitHub Release should contain it.

This is the exact file used for the AssayLoop results. Unlike older DepMap
releases deposited on Figshare under CC BY 4.0, newly generated DepMap data is
subject to the portal's current terms. Those terms permit research use and
some non-profit research sharing, but restrict clinical and commercial use.
They also require anyone rehosting the data to repost the terms in full and
require downstream users to adhere to them.

AssayBench's MIT license covers the software, not this user-supplied file. The
third-party notice is in `THIRD_PARTY_NOTICES.md`. The authoritative terms are
at https://depmap.org/portal/terms/ and should be checked for updates.

## Citation

> DepMap, Broad. *DepMap Public 26Q1*. Dataset. https://depmap.org/portal/
