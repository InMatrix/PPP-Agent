"""Small compatibility aliases across supported Transformers generations."""

from transformers import AutoModelForImageTextToText

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    # Transformers 5 removed the deprecated name. Its replacement has the
    # same auto-model role and mapping contract used by Verl.
    AutoModelForVision2Seq = AutoModelForImageTextToText


__all__ = ["AutoModelForVision2Seq"]
