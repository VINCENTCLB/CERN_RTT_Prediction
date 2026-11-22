from pathlib import Path

from processing.flow_data import FlowData
from processing.ip_addresses import IpPrefixes


class FeatureBuilder:
    def __init__(self, root):
        self.root = root
        self.data = FlowData(root)

    def build(self):
        # Make prefixes once per address instead of once per flow.
        addresses = self.data.query("""
            SELECT DISTINCT ip FROM (
                SELECT "meta.src_ip" AS ip FROM flows
                UNION
                SELECT "meta.dst_ip" FROM flows
            ) WHERE ip IS NOT NULL
        """)["ip"].tolist()
        lookup = IpPrefixes().table(addresses)
        # DuckDB can join this table without saving another file.
        self.data.con.register("ip_lookup", lookup)

        output = self.root / "data" / "features.parquet"
        output.parent.mkdir(exist_ok=True)
        split = self.data.split_label()
        # Keep the target as a label and leave out end-event fields.
        # COPY writes all rows without building a full pandas table.
        self.data.con.execute(f"""
            COPY (
                SELECT
                    delta_rtt,
                    start_rtt,
                    {split} AS split,
                    date_part('hour', cast(flow_start AS TIMESTAMPTZ)) AS hour_utc,
                    date_part('dow', cast(flow_start AS TIMESTAMPTZ)) AS day_of_week_utc,
                    src.broad AS src_broad,
                    src.medium AS src_medium,
                    src.narrow AS src_narrow,
                    dst.broad AS dst_broad,
                    dst.medium AS dst_medium,
                    dst.narrow AS dst_narrow,
                    "meta.src_port",
                    "meta.dst_port",
                    "meta.cric.src.name",
                    "meta.cric.dst.name",
                    "meta.cric.src.asn",
                    "meta.cric.dst.asn",
                    "meta.application",
                    "meta.experiment_name",
                    "meta.activity_id",
                    "meta.ip_version",
                    "meta.syslog.hostname"
                FROM flows AS flow
                LEFT JOIN ip_lookup AS src ON src.ip = flow."meta.src_ip"
                LEFT JOIN ip_lookup AS dst ON dst.ip = flow."meta.dst_ip"
            ) TO '{output.as_posix()}' (FORMAT PARQUET)
        """)
        print("saved", output)


if __name__ == "__main__":
    FeatureBuilder(Path.cwd()).build()
