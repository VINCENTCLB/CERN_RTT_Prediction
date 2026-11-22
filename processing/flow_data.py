import json

import duckdb


class FlowData:
    def __init__(self, root):
        self.path = root / "student_dataset" / "flows.parquet"
        self.splits = json.loads((root / "student_dataset" / "splits.json").read_text())
        self.con = duckdb.connect()
        # DuckDB uses UTC when it extracts hours and weekdays.
        self.con.execute("SET TimeZone='UTC'")
        # The view reads Parquet rows when a query needs them.
        self.con.execute(f"CREATE VIEW flows AS SELECT * FROM read_parquet('{self.path.as_posix()}')")

    def query(self, sql):
        return self.con.execute(sql).fetchdf()

    def count(self):
        return self.query("SELECT count(*) AS n FROM flows").n[0]

    def size_gb(self):
        return self.path.stat().st_size / 1e9

    def split_label(self):
        train_end = self.splits["train_end_inclusive"]
        val_end = self.splits["validation_end_inclusive"]
        # Parse offsets so equal UTC times get the same split.
        return (
            f"CASE WHEN cast(flow_start AS TIMESTAMPTZ) <= TIMESTAMPTZ '{train_end}' THEN 'train' "
            f"WHEN cast(flow_start AS TIMESTAMPTZ) <= TIMESTAMPTZ '{val_end}' THEN 'validation' "
            f"ELSE 'test' END"
        )

    def expected_counts(self):
        counts = {}
        for name, info in self.splits["splits"].items():
            counts[name] = info["row_count"]
        return counts
