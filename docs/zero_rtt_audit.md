# Zero end-RTT audit

This audit counts flows with `end_rtt = 0` by split, site and application. The data provider advised treating these values as suspect. The script reports exclusion candidates; it does not modify or filter the dataset.

## Run

Use Python 3.11+ and DuckDB 2.0.0. From the repository root:

```bash
python -m pip install duckdb==2.0.0
python processing/zero_rtt_audit.py --dataset-dir student_dataset --output-dir data/zero_rtt_audit
python -m unittest discover -s tests -p 'test_zero_rtt_audit.py'
```

The input directory must contain `flows.parquet`, `manifest.json` and `splits.json`. The output directory must be outside the input directory. The repository ignores `data/` and Parquet files.

## Release 1 results

Counts use the original chronological split, before any other filtering:

| Split | Rows | Zero end RTT | Rows after exclusion |
|---|---:|---:|---:|
| Train | 47,071,163 | 340,142 | 46,731,021 |
| Validation | 10,086,678 | 17,135 | 10,069,543 |
| Test | 10,086,678 | 185 | 10,086,493 |
| Total | 67,244,519 | 357,462 | 66,887,057 |

The audit produces six CSVs with split totals and source-site, destination-site and application breakdowns. Fractions use the original count in each group. NULL values are distinguished from blanks and `unknown`.

Site labels are from the start event and do not identify client/server roles. The release contains no final byte counters, so the provider's suspected cause cannot be checked here. End-event fields and exclusion flags must not be model inputs at flow start. A negative `delta_rtt` alone is not an exclusion. The counts above do not account for labels unavailable at the training cutoff; RTT units remain unverified.
