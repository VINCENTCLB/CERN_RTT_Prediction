#!/usr/bin/env python3
"""Report exact-zero final RTT observations in the Firefly release."""

import argparse
import csv
import json
from pathlib import Path

import duckdb


def sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True,
                        help='Directory containing flows.parquet, splits.json and manifest.json.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Separate directory for aggregate CSV reports.')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == dataset_dir or dataset_dir in output_dir.parents:
        parser.error('Choose an output directory outside the original dataset directory.')
    parquet_path = dataset_dir / 'flows.parquet'
    split_config = json.loads((dataset_dir / 'splits.json').read_text(encoding='utf-8'))
    manifest = json.loads((dataset_dir / 'manifest.json').read_text(encoding='utf-8'))
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute("SET TimeZone = 'UTC'")
    connection.execute('SET threads = 4')
    connection.execute("SET memory_limit = '2GB'")

    # Distinguish NULL from blank strings and the literal category "unknown".
    query = f"""
        WITH records AS (
            SELECT TRY_CAST(flow_start AS TIMESTAMPTZ) AS start_time,
                   end_rtt, start_rtt, delta_rtt,
                   "meta.cric.src.name" AS src_site,
                   "meta.cric.dst.name" AS dst_site,
                   "meta.application" AS application
            FROM read_parquet({sql_literal(parquet_path)})
        ), tagged AS (
            SELECT CASE
                       WHEN start_time IS NULL THEN 'invalid_start'
                       WHEN start_time <= TIMESTAMPTZ {sql_literal(split_config['train_end_inclusive'])}
                           THEN 'train'
                       WHEN start_time <= TIMESTAMPTZ {sql_literal(split_config['validation_end_inclusive'])}
                           THEN 'validation'
                       ELSE 'test'
                   END AS split,
                   src_site, dst_site, application, start_rtt, end_rtt, delta_rtt
            FROM records
        )
        SELECT split, src_site, dst_site, application,
               GROUPING(src_site) AS src_is_grouped_away,
               GROUPING(dst_site) AS dst_is_grouped_away,
               GROUPING(application) AS application_is_grouped_away,
               COUNT(*) AS rows_before,
               COUNT(*) FILTER (WHERE end_rtt = 0) AS zero_end_rtt,
               COUNT(*) FILTER (WHERE end_rtt = 0 AND start_rtt > 0) AS zero_end_positive_start,
               COUNT(*) FILTER (WHERE end_rtt IS NULL) AS null_end_rtt,
               COUNT(*) FILTER (WHERE delta_rtt < 0) AS negative_delta,
               COUNT(*) FILTER (WHERE delta_rtt < 0 AND end_rtt = 0) AS negative_delta_with_zero_end
        FROM tagged
        GROUP BY GROUPING SETS (
            (split),
            (split, src_site),
            (split, dst_site),
            (split, application),
            (split, src_site, application),
            (split, dst_site, application)
        )
    """
    cursor = connection.execute(query)
    names = [item[0] for item in cursor.description]
    rows = [dict(zip(names, values)) for values in cursor.fetchall()]
    connection.close()

    grouping_names = {
        (1, 1, 1): 'split',
        (0, 1, 1): 'src_site',
        (1, 0, 1): 'dst_site',
        (1, 1, 0): 'application',
        (0, 1, 0): 'src_site_application',
        (1, 0, 0): 'dst_site_application',
    }
    buckets = {name: [] for name in grouping_names.values()}
    for row in rows:
        key = (row.pop('src_is_grouped_away'), row.pop('dst_is_grouped_away'),
               row.pop('application_is_grouped_away'))
        group = grouping_names[key]
        for category in ('src_site', 'dst_site', 'application'):
            row[category + '_is_null'] = row[category] is None if category in group else ''
            if row[category] is None:
                row[category] = ''
        row['rows_after_excluding_zero_end'] = row['rows_before'] - row['zero_end_rtt']
        row['zero_end_fraction'] = row['zero_end_rtt'] / row['rows_before']
        buckets[group].append(row)

    splits = {row['split']: row for row in buckets['split']}
    expected_splits = set(split_config['splits'])
    if set(splits) != expected_splits:
        raise ValueError(f'Unexpected or unparseable split assignments: {sorted(splits)}')
    for split in expected_splits:
        if splits[split]['rows_before'] != split_config['splits'][split]['row_count']:
            raise ValueError(f'Split count does not match splits.json: {split}')
    if sum(row['rows_before'] for row in buckets['split']) != manifest['row_count']:
        raise ValueError('Row count does not match manifest.json')
    for group, values in buckets.items():
        for split in expected_splits:
            subset = [row for row in values if row['split'] == split]
            for field in ('rows_before', 'zero_end_rtt', 'zero_end_positive_start',
                          'null_end_rtt', 'negative_delta', 'negative_delta_with_zero_end'):
                if sum(row[field] for row in subset) != splits[split][field]:
                    raise ValueError(f'{group}, {split}: {field} does not reconcile')

    columns = ['split', 'src_site', 'src_site_is_null', 'dst_site', 'dst_site_is_null',
               'application', 'application_is_null', 'rows_before', 'zero_end_rtt',
               'rows_after_excluding_zero_end', 'zero_end_fraction',
               'zero_end_positive_start', 'null_end_rtt', 'negative_delta',
               'negative_delta_with_zero_end']
    for group, values in buckets.items():
        destination = output_dir / f'{group}.csv'
        with destination.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(sorted(values, key=lambda row: (row['split'], -row['zero_end_rtt'],
                                                               row['src_site'], row['dst_site'], row['application'])))

    print('Exact-zero end RTT audit; source files were not modified.')
    for split in ('train', 'validation', 'test'):
        row = splits[split]
        print(f"{split:10} before={row['rows_before']:>10,}  zero={row['zero_end_rtt']:>8,}  "
              f"after={row['rows_after_excluding_zero_end']:>10,}  "
              f"zero%={row['zero_end_fraction'] * 100:.4f}")
    print('All totals reconcile with release metadata and every grouped report.')


if __name__ == '__main__':
    main()
