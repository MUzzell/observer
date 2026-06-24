"""Parse the capture timestamp embedded in camera clip filenames.

The camera Pi names clips with a ``YYYYMMDDHHMMSS`` stamp, e.g.
``0-141-20260327132601.mp4`` -> 2026-03-27 13:26:01. Other sources (phone videos
named with a UUID) have no stamp and return ``None``.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

_TS = re.compile(r"(\d{14})")


def parse_capture_time(name: str) -> Optional[datetime]:
    """Return the capture time from a filename, or None if there isn't one."""
    matches = _TS.findall(name)
    if not matches:
        return None
    # The stamp is at the end of the name; take the last 14-digit run.
    try:
        return datetime.strptime(matches[-1], "%Y%m%d%H%M%S")
    except ValueError:
        return None
