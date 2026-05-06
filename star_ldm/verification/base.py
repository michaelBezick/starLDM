from typing import Optional


class BaseVerifier:
    def verify(self, prompt: str, decoded: str, gold: Optional[str] = None) -> bool:
        raise NotImplementedError
