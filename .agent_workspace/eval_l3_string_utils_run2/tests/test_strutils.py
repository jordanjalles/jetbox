import unittest
from strutils import capitalize, reverse, count_words

class TestStrutils(unittest.TestCase):
    def test_capitalize(self):
        self.assertEqual(capitalize('hello'), 'Hello')
        self.assertEqual(capitalize('hELLO'), 'Hello')
        self.assertEqual(capitalize(''), '')

    def test_reverse(self):
        self.assertEqual(reverse('abc'), 'cba')
        self.assertEqual(reverse(''), '')

    def test_count_words(self):
        self.assertEqual(count_words('Hello world'), 2)
        self.assertEqual(count_words('  multiple   spaces  '), 2)
        self.assertEqual(count_words(''), 0)

if __name__ == '__main__':
    unittest.main()
