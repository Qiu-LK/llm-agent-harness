import unittest

from src.service import greet


class GreetingTests(unittest.TestCase):
    def test_greet(self) -> None:
        self.assertEqual(greet("Harness"), "Hello, Harness!")


    def test_greet_blank_input_uses_fallback(self) -> None:
        self.assertEqual(greet(""), "Hello, qiuqiuqiu!")
        self.assertEqual(greet("   "), "Hello, qiuqiuqiu!")


    def test_greet_strips_surrounding_whitespace(self) -> None:
        self.assertEqual(greet("  Bob  "), "Hello, Bob!")
