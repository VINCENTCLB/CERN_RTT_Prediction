import unittest
from pathlib import Path

from processing.flow_data import FlowData
from processing.ip_addresses import IpPrefixes


ROOT = Path(__file__).resolve().parents[1]


class ProcessingTests(unittest.TestCase):
    def test_prefixes(self):
        prefixes = IpPrefixes()
        self.assertEqual(
            prefixes.for_address("192.0.2.10"),
            (4, "192.0.0.0/16", "192.0.2.0/24", "192.0.2.0/24"),
        )
        self.assertEqual(
            prefixes.for_address("2001:db8:1:2::1"),
            (6, "2001:db8::/32", "2001:db8:1::/48", "2001:db8:1:2::/64"),
        )

    def test_split_boundaries(self):
        data = FlowData(ROOT)
        rows = data.query(f"""
            SELECT {data.split_label()} AS split
            FROM (VALUES
                ('2026-10-12T13:12:05.477960-04:00'),
                ('2026-10-12T17:12:05.481692+00:00'),
                ('2026-10-13T22:18:28.700870+00:00'),
                ('2026-10-13T22:18:28.701549+00:00')
            ) AS times(flow_start)
        """)
        self.assertEqual(rows["split"].tolist(), ["train", "validation", "validation", "test"])


if __name__ == "__main__":
    unittest.main()
