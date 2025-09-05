import unittest
from src.data.processing import process_data

class TestProcessing(unittest.TestCase):
    def test_process_data_returns_chunks(self):
        chunks = process_data()
        self.assertIsNotNone(chunks)
        self.assertTrue(len(chunks) > 0)

if __name__ == "__main__":
    unittest.main()