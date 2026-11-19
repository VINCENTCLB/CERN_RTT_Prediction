import ipaddress

import pandas as pd
"""
Turns IP strings into network prefixes, so the model can group flows by network instead of by individual machines.

    prefixes = IpPrefixes()
    lookup = prefixes.table(flows.query("SELECT DISTINCT src_ip FROM flows").src_ip)

    example of lookup table:
                          ip  version           broad               medium                  narrow
0  2620:0:28b0:64::8037:4c6e        6       2620::/32     2620:0:28b0::/48     2620:0:28b0:64::/64
1 2a00:1398:4:1847::ac1e:156        6  2a00:1398::/32     2a00:1398:4::/48   2a00:1398:4:1847::/64
2   2001:1458:303:1a::100:2b        6  2001:1458::/32   2001:1458:303::/48   2001:1458:303:1a::/64
3   2001:1470:ff8a:6d:d14::5        6  2001:1470::/32  2001:1470:ff8a::/48  2001:1470:ff8a:6d::/64
4             193.205.66.145        4  193.205.0.0/16      193.205.66.0/24         193.205.66.0/24
5   2001:1458:301:10::100:96        6  2001:1458::/32   2001:1458:301::/48   2001:1458:301:10::/64
"""

class IpPrefixes:
    def __init__(self):
        self.bits = {4: [16, 24, 24], 6: [32, 48, 64]}

    def for_address(self, address):
        ip = ipaddress.ip_address(address)
        prefixes = []
        for size in self.bits[ip.version]:# IPv4 or IPv6 and pick the corresponding size
            # ip_network("192.0.2.10/24") raises ValueError for having host bits set
            # ip_network("192.0.2.10/24", strict=False) or ("192.0.2.0/24") are ok
            network = ipaddress.ip_network(f"{address}/{size}", strict=False)
            # store as text
            prefixes.append(str(network))
         # order matches self.bits: broad, medium, narrow.
        return ip.version, prefixes[0], prefixes[1], prefixes[2]

    def table(self, addresses):
        rows = []
        for address in addresses:
            version, broad, medium, narrow = self.for_address(address)
            rows.append([address, version, broad, medium, narrow])
        return pd.DataFrame(rows, columns=["ip", "version", "broad", "medium", "narrow"])
