import unittest

class TestLevel2(unittest.TestCase):
    def test_agent_type_a(self):
        self.assertTrue(True)
    def test_agent_type_b(self):
        self.assertEqual('agent', 'agent')
    def test_agent_type_c(self):
        # Intentional failure
        self.assertEqual(1, 2)
    def test_agent_type_d(self):
        self.assertIsInstance(5, int)
    def test_agent_type_e(self):
        self.assertFalse(False)
