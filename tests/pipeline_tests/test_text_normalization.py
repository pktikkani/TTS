import unittest

from scripts.text.normalize_en import normalize_for_tts


class TestEnglishNormalization(unittest.TestCase):
    def test_normalizes_currency_and_time(self):
        text = "The invoice total is $149.50 and payment is due by 5:30 PM."
        normalized = normalize_for_tts(text)
        self.assertIn("one hundred forty nine dollars and fifty cents", normalized)
        self.assertIn("five thirty p m", normalized)

    def test_normalizes_date_and_acronyms(self):
        text = "The API review is on 04/15/2026 for the GPU team."
        normalized = normalize_for_tts(text)
        self.assertIn("a p i", normalized)
        self.assertIn("April fifteenth two thousand twenty six", normalized)
        self.assertIn("g p u", normalized)

    def test_normalizes_email_and_url(self):
        text = "Email support@example.com or visit https://openai.com/docs."
        normalized = normalize_for_tts(text)
        self.assertIn("support at example dot com", normalized)
        self.assertIn("openai dot com slash docs", normalized)


if __name__ == "__main__":
    unittest.main()
