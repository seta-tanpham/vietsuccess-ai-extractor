"""Phase 1 — Step 3: regex-based filler word removal (no LLM)."""
import re

# Vietnamese + English filler patterns — single words and phrases
_FILLERS = [
    # Single hesitations
    r"\b(ừ|à|ờ|ơ|uh|um|hmm|ừm|mhm|er|erm)\b",
    # Hedging phrases
    r"\b(kiểu như|kiểu là|cái là|đại loại|basically|you know|i mean|right)\b",
    r"\bthì\b(?=\s)",
    # Redundant openers
    r"\b(thì cái vấn đề ở đây là|như là|cái chỗ này là|so basically|so like)\b",
]

_PATTERN = re.compile("|".join(_FILLERS), re.IGNORECASE | re.UNICODE)
_MULTI_SPACE = re.compile(r" {2,}")


def clean_text(text: str) -> str:
    """Remove filler words, preserve content words. Returns cleaned string."""
    cleaned = _PATTERN.sub(" ", text)
    cleaned = _MULTI_SPACE.sub(" ", cleaned).strip()
    return cleaned


def compression_ratio(original: str, cleaned: str) -> float:
    """Return length reduction ratio. Should stay between 0.05–0.15."""
    if not original:
        return 0.0
    return 1 - len(cleaned) / len(original)
