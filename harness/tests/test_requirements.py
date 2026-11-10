import unittest

from harness.requirements import SKILLS, load_domain, parse_requirements


class TestRequirements(unittest.TestCase):
    def test_cars_ladder(self):
        d = load_domain("cars")
        self.assertEqual(len(d.skills), 10)
        self.assertEqual([s.name for s in d.skills], list(SKILLS))
        s3 = d.skills[2]
        self.assertEqual((s3.index, s3.name), (3, "certify"))
        self.assertEqual([r.id for r in s3.requests], ["CAR-05", "CAR-06"])
        self.assertTrue(s3.requests[0].text.startswith("Compare the fuel efficiency"))
        self.assertTrue(d.csv_path.name == "auto93.csv" and d.csv_path.exists())

    def test_nasa_and_apache(self):
        self.assertEqual([r.id for r in load_domain("nasa").skills[0].requests], ["NASA-01", "NASA-02"])
        self.assertEqual(sum(len(s.requests) for s in load_domain("apache").skills), 20)

    def test_bad_document(self):
        with self.assertRaises(ValueError):
            parse_requirements("### 1. Remember — Representation\n\n**CAR-01:** only one\n")


if __name__ == "__main__":
    unittest.main()
