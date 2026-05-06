import re
from decimal import Decimal, InvalidOperation
from typing import Optional

from star_ldm.verification.base import BaseVerifier


_NUMBER_PATTERN = r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'


def _to_decimal(value: str) -> Optional[Decimal]:
    value = value.strip().replace(',', '')
    if not value:
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def _format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, 'f').rstrip('0').rstrip('.')


def extract_gsm8k_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    patterns = [
        rf'####\s*({_NUMBER_PATTERN})',
        rf'(?:final answer|answer|therefore|so)\s*(?:is|=|:)?\s*\$?\s*({_NUMBER_PATTERN})',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            value = _to_decimal(matches[-1])
            if value is not None:
                return _format_decimal(value)

    matches = re.findall(_NUMBER_PATTERN, text)
    if not matches:
        return None
    value = _to_decimal(matches[-1])
    if value is None:
        return None
    return _format_decimal(value)


class GSM8KVerifier(BaseVerifier):
    def extract_answer(self, text: Optional[str]) -> Optional[str]:
        return extract_gsm8k_answer(text)

    def verify(self, prompt: str, decoded: str, gold: Optional[str] = None) -> bool:
        gold_answer = self.extract_answer(gold)
        decoded_answer = self.extract_answer(decoded)
        if gold_answer is None or decoded_answer is None:
            return False
        return gold_answer == decoded_answer
