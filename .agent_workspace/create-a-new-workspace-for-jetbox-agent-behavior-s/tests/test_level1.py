import unittest

class TestLevel1(unittest.TestCase):
    def test_behavior_a(self):
        self.assertTrue(True)
    def test_behavior_b(self):
        self.assertEqual(1+1, 2)
    def test_behavior_c(self):
        self.assertIn('a', 'abc')
    def test_behavior_d(self):
        self.assertIsNotNone(42)
    def test_behavior_e(self):
        self.assertGreater(10, 5)
