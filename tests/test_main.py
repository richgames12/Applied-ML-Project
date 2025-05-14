import unittest
from main import hello_world


class MainTest(unittest.TestCase):
    def test_hello(self):
        self.assertEqual(hello_world(), "Hello, World!, Made a test")


if __name__ == '__main__':
    unittest.main()
