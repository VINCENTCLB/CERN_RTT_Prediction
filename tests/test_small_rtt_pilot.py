import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

spec = importlib.util.spec_from_file_location('pilot', Path(__file__).resolve().parents[1] / 'processing' / 'small_rtt_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotTests(unittest.TestCase):
    def test_output_cannot_overwrite_source_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, str(spec.origin), '--dataset-dir', str(root),
                       '--output', str(root / 'flows.parquet')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('outside the original dataset directory', result.stderr)

    def test_sampling_respects_cutoffs_and_zero_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'flows.parquet'
            con = duckdb.connect()
            con.execute("SET TimeZone = 'UTC'")
            con.execute('''CREATE TABLE flows (flow_id VARCHAR, flow_start VARCHAR, flow_end VARCHAR,
                start_rtt DOUBLE, end_rtt DOUBLE, delta_rtt DOUBLE,
                "meta.application" VARCHAR, "meta.experiment_name" VARCHAR,
                "meta.activity_id" BIGINT, "meta.ip_version" BIGINT)''')
            rows = [
                ('a','2026-01-01T12:00:00+00:00','2026-01-01T13:00:00+00:00',4,2,-2,'a','cms',1,6),
                ('b','2026-01-01T12:00:00+00:00','2026-01-03T00:00:00+00:00',4,2,-2,'a','cms',1,6),
                ('c','2026-01-01T12:00:00+00:00','2026-01-01T13:00:00+00:00',4,0,-4,'a','cms',1,6),
                ('d','2026-01-03T12:00:00+00:00','2026-01-03T13:00:00+00:00',4,1,-3,'b','cms',1,6),
                ('e','2026-01-03T12:00:00+00:00','2026-01-05T00:00:00+00:00',4,2,-2,'b','cms',1,6),
                ('f','2026-01-05T12:00:00+00:00','2026-01-05T13:00:00+00:00',4,2,-2,'a','cms',1,6),
            ]
            con.executemany('INSERT INTO flows VALUES (?,?,?,?,?,?,?,?,?,?)', rows)
            con.execute(f"COPY flows TO {pilot.quoted(p)} (FORMAT PARQUET)")
            splits = {'train_end_inclusive':'2026-01-02T00:00:00+00:00',
                      'validation_end_inclusive':'2026-01-04T00:00:00+00:00'}
            self.assertEqual(pilot.sample(con,p,splits,'train',100,42)['delta_rtt'].tolist(), [-2.0])
            self.assertEqual(pilot.sample(con,p,splits,'validation',100,42)['delta_rtt'].tolist(), [-3.0])
            con.close()

    def test_unseen_and_missing_categories_are_distinct(self):
        self.assertIsNone(pilot.key(pd.NA))
        train = pd.DataFrame({'start_rtt':[1.,2.], 'hour_utc':[1,2], 'dow_utc':[1,2],
                              'application':['known',None], 'experiment':['cms','cms'],
                              'activity':[1,1], 'ip_version':[6,6]})
        val = train.iloc[[0]].copy()
        val['application'] = 'unseen'
        X, V = pilot.fit_transform(train,val)
        self.assertEqual(X[1,5], 1)  # missing slot
        self.assertEqual(V[0,6], 1)  # unseen slot


if __name__ == '__main__':
    unittest.main()
