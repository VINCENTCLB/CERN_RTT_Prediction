"""Small synthetic-data check for the zero end-RTT audit."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb


SCRIPT = Path(__file__).resolve().parents[1] / 'processing' / 'zero_rtt_audit.py'


class ZeroRttAuditTest(unittest.TestCase):
    def test_split_boundaries_null_groups_and_exact_zero(self):
        # Offset timestamp equals the inclusive UTC train boundary.
        records = [
            ('2026-10-12T13:12:05.477960-04:00', 10.0, 0.0, -10.0, 'A', None, 'xrdcp'),
            ('2026-10-12T17:12:05.477960+00:00', 0.0, -0.0, 0.0, None, '', 'unknown'),
            ('2026-10-12T17:12:05.000000+00:00', 2.0, 1.0, -1.0, 'A', 'B', None),
            ('2026-10-12T17:12:05.477961+00:00', 1.0, 2.0, 1.0, 'A', 'B', 'xrdcp'),
            ('2026-10-13T22:18:28.700870+00:00', 2.0, 0.0, -2.0, 'A', 'B', 'xrdcp'),
            ('2026-10-13T22:18:28.700871+00:00', 3.0, 0.0, -3.0, 'A', 'B', 'xrdcp'),
            ('2026-10-14T12:00:00.000000+00:00', 3.0, None, None, None, 'B', None),
            ('2026-10-14T13:00:00.000000+00:00', 3.0, 2.0, -1.0, 'A', 'B', 'xrdcp'),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / 'student_dataset'
            report_dir = root / 'reports'
            data_dir.mkdir()
            connection = duckdb.connect()
            connection.execute('''CREATE TABLE flows (
                flow_start VARCHAR, start_rtt DOUBLE, end_rtt DOUBLE, delta_rtt DOUBLE,
                "meta.cric.src.name" VARCHAR, "meta.cric.dst.name" VARCHAR,
                "meta.application" VARCHAR)''')
            connection.executemany('INSERT INTO flows VALUES (?, ?, ?, ?, ?, ?, ?)', records)
            path = str(data_dir / 'flows.parquet').replace("'", "''")
            connection.execute(f"COPY flows TO '{path}' (FORMAT PARQUET)")
            connection.close()
            (data_dir / 'manifest.json').write_text(json.dumps({'row_count': len(records)}))
            (data_dir / 'splits.json').write_text(json.dumps({
                'train_end_inclusive': '2026-10-12T17:12:05.477960+00:00',
                'validation_end_inclusive': '2026-10-13T22:18:28.700870+00:00',
                'splits': {'train': {'row_count': 3}, 'validation': {'row_count': 2},
                           'test': {'row_count': 3}},
            }))
            result = subprocess.run([sys.executable, str(SCRIPT), '--dataset-dir', str(data_dir),
                                     '--output-dir', str(report_dir)],
                                    capture_output=True, text=True, check=True)
            self.assertIn('All totals reconcile', result.stdout)
            with (report_dir / 'split.csv').open(newline='') as stream:
                splits = {row['split']: row for row in csv.DictReader(stream)}
            self.assertEqual([int(splits[s]['zero_end_rtt']) for s in ('train', 'validation', 'test')], [2, 1, 1])
            self.assertEqual([int(splits[s]['rows_after_excluding_zero_end']) for s in ('train', 'validation', 'test')], [1, 1, 2])
            self.assertEqual(int(splits['test']['null_end_rtt']), 1)
            self.assertEqual(int(splits['train']['negative_delta']), 2)
            self.assertEqual(int(splits['train']['negative_delta_with_zero_end']), 1)
            with (report_dir / 'application.csv').open(newline='') as stream:
                categories = list(csv.DictReader(stream))
            train_categories = [row for row in categories if row['split'] == 'train']
            self.assertTrue(any(r['application_is_null'] == 'True' for r in train_categories))
            self.assertTrue(any(r['application'] == 'unknown' and r['application_is_null'] == 'False' for r in train_categories))


if __name__ == '__main__':
    unittest.main()
