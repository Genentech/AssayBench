# Third-party data notices

The MIT License in `LICENSE` covers AssayBench software and documentation
owned by Genentech. It does not grant rights to third-party datasets that
AssayBench can reference or operate on.

## DepMap Public 26Q1

AssayBench does not host or download `CRISPRInferredCommonEssentials.csv`.
Users who invoke the default `percent_essential` metric must obtain the file
directly from DepMap and provide its local path. The file is not included in
the current source tree, source distribution, or wheel. It remains subject to
the DepMap Terms of Use, including its research-purpose limitation and
restrictions on clinical and commercial use. By downloading or using the file,
you must comply with those terms.

The authoritative current terms are at https://depmap.org/portal/terms/ and
should be reviewed when obtaining or using the data.

Source and attribution:

> DepMap, Broad. *DepMap Public 26Q1*. Dataset.
> https://depmap.org/portal/

Exact-file provenance and checksums are recorded in
`src/assaybench/data/depmap/PROVENANCE.md`.

Other external assets and their applicable terms are listed by:

```console
assaybench assets
```
