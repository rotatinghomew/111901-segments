import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class FixtureIntegrityTest(unittest.TestCase):
    def test_csv_files_have_headers_and_records(self):
        files = sorted((ROOT / "data").glob("*.csv"))
        self.assertGreaterEqual(len(files), 2)
        for path in files:
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows, path.name)
            self.assertTrue(all(any(value for value in row.values()) for row in rows), path.name)

    def test_policy_json_is_valid_when_present(self):
        for path in (ROOT / "data").glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsInstance(value, dict, path.name)

if __name__ == "__main__":
    unittest.main()
