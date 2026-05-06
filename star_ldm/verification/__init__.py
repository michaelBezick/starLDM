from star_ldm.verification.base import BaseVerifier
from star_ldm.verification.exact_match import ExactMatchVerifier


_VERIFIERS = {
    'exact_match': ExactMatchVerifier,
}


def get_verifier(name):
    try:
        return _VERIFIERS[name]()
    except KeyError as exc:
        available = ', '.join(sorted(_VERIFIERS))
        raise ValueError(f'Unknown verifier {name!r}. Available: {available}') from exc


__all__ = ['BaseVerifier', 'ExactMatchVerifier', 'get_verifier']
