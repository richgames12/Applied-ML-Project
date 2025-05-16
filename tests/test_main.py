import unittest


class MainTest(unittest.TestCase):
    def test_hello(self):
        self.assertEqual("Hello, World!", "Hello, World!")


if __name__ == '__main__':
    unittest.main()
