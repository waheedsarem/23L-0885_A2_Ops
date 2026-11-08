import unittest


class ArithmeticTest(unittest.TestCase):
    def test_addition(self):
        # Deliberate first red run; corrected in the following commit.
        self.assertEqual(2 + 2, 5)


if __name__ == "__main__":
    unittest.main()
