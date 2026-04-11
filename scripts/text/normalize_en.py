#!/usr/bin/env python3
"""English text normalization for product-facing TTS inputs."""

from __future__ import annotations

import re


ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}

ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth",
    30: "thirtieth",
}

MONTHS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

ABBREVIATIONS = {
    "Mr.": "Mister",
    "Mrs.": "Misses",
    "Ms.": "Miss",
    "Dr.": "Doctor",
    "Prof.": "Professor",
}


def number_to_words(number: int) -> str:
    if number < 0:
        return f"minus {number_to_words(abs(number))}"
    if number < 20:
        return ONES[number]
    if number < 100:
        tens = (number // 10) * 10
        rest = number % 10
        return TENS[tens] if rest == 0 else f"{TENS[tens]} {ONES[rest]}"
    if number < 1000:
        hundreds = number // 100
        rest = number % 100
        return f"{ONES[hundreds]} hundred" if rest == 0 else f"{ONES[hundreds]} hundred {number_to_words(rest)}"
    if number < 1_000_000:
        thousands = number // 1000
        rest = number % 1000
        prefix = f"{number_to_words(thousands)} thousand"
        return prefix if rest == 0 else f"{prefix} {number_to_words(rest)}"
    if number < 1_000_000_000:
        millions = number // 1_000_000
        rest = number % 1_000_000
        prefix = f"{number_to_words(millions)} million"
        return prefix if rest == 0 else f"{prefix} {number_to_words(rest)}"
    billions = number // 1_000_000_000
    rest = number % 1_000_000_000
    prefix = f"{number_to_words(billions)} billion"
    return prefix if rest == 0 else f"{prefix} {number_to_words(rest)}"


def ordinal_to_words(number: int) -> str:
    if number in ORDINALS:
        return ORDINALS[number]
    if number < 100:
        tens = (number // 10) * 10
        rest = number % 10
        return f"{TENS[tens]} {ORDINALS[rest]}"
    if number < 1000:
        hundreds = number // 100
        rest = number % 100
        prefix = f"{ONES[hundreds]} hundred"
        return f"{prefix} {ordinal_to_words(rest)}"
    return f"{number_to_words(number)}th"


def _normalize_abbreviations(text: str) -> str:
    for raw, spoken in ABBREVIATIONS.items():
        text = text.replace(raw, spoken)
    return text


def _normalize_currency(match: re.Match[str]) -> str:
    dollars = int(match.group("dollars"))
    cents = match.group("cents")
    parts = [f"{number_to_words(dollars)} dollars"]
    if cents:
        parts.append(f"{number_to_words(int(cents))} cents")
    return " and ".join(parts)


def _normalize_time(match: re.Match[str]) -> str:
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    suffix = match.group("suffix")
    minute_words = "o'clock" if minute == 0 else number_to_words(minute)
    if minute < 10 and minute != 0:
        minute_words = f"oh {ONES[minute]}"
    spoken = f"{number_to_words(hour)} {minute_words}"
    if suffix:
        spoken = f"{spoken} {' '.join(suffix.lower())}"
    return spoken


def _normalize_date(match: re.Match[str]) -> str:
    month = int(match.group("month"))
    day = int(match.group("day"))
    year = int(match.group("year"))
    month_name = MONTHS.get(month, str(month))
    return f"{month_name} {ordinal_to_words(day)} {number_to_words(year)}"


def _spell_token(token: str) -> str:
    token = token.lower()
    token = token.replace(".", " dot ")
    token = token.replace("-", " dash ")
    token = token.replace("_", " underscore ")
    token = token.replace("/", " slash ")
    return " ".join(token.split())


def _normalize_email(match: re.Match[str]) -> str:
    local, domain = match.group(0).split("@", 1)
    return f"{_spell_token(local)} at {_spell_token(domain)}"


def _normalize_url(match: re.Match[str]) -> str:
    value = match.group(0)
    value = re.sub(r"^https?://", "", value)
    return _spell_token(value)


def _normalize_acronyms(text: str) -> str:
    return re.sub(r"\b([A-Z]{2,})\b", lambda match: " ".join(match.group(1).lower()), text)


def _normalize_numbers(text: str) -> str:
    return re.sub(r"\b\d+\b", lambda match: number_to_words(int(match.group(0))), text)


def normalize_for_tts(text: str) -> str:
    text = _normalize_abbreviations(text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", _normalize_email, text)
    text = re.sub(r"https?://[^\s]+|www\.[^\s]+", _normalize_url, text)
    text = re.sub(
        r"\$(?P<dollars>\d+)(?:\.(?P<cents>\d{2}))?",
        _normalize_currency,
        text,
    )
    text = re.sub(
        r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})\b",
        _normalize_date,
        text,
    )
    text = re.sub(
        r"\b(?P<hour>\d{1,2}):(?P<minute>\d{2})(?:\s?(?P<suffix>AM|PM|am|pm))?\b",
        _normalize_time,
        text,
    )
    text = _normalize_acronyms(text)
    text = _normalize_numbers(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
