import re
from typing import Optional

from star_ldm.verification.base import BaseVerifier


class ExactMatchVerifier(BaseVerifier):
    def normalize(self, text: Optional[str]) -> str:
        if text is None:
            return ''
        return re.sub(r'\s+', ' ', text).strip().lower()

    def verify(self, prompt: str, decoded: str, gold: Optional[str] = None) -> bool:
        if gold is None:
            return False
        return self.normalize(decoded) == self.normalize(gold)
