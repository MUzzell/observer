"""Process-wide logging configuration.

Python's default only surfaces ``WARNING``+ through a last-resort handler, so the
``observer.*`` ``INFO`` logs (worker lifecycle, per-clip processing, detector
backend errors) never reach the journal. Call :func:`setup_logging` once at
process start to stream everything to stdout — which ``systemd`` captures, so
``journalctl -u observer -f`` shows it — and, when a data directory is known, to
a rotating file you can ``tail -f`` directly.

The level is ``INFO`` by default; override with ``OBSERVER_LOG_LEVEL`` (e.g.
``OBSERVER_LOG_LEVEL=DEBUG``).

Idempotent and self-healing: it tags the handlers it installs, so calling it
again (e.g. once in the CLI and once in the FastAPI lifespan) won't duplicate
them, and it re-installs the stdout handler if something — such as uvicorn's own
``dictConfig`` — has since wiped the root handlers.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_MARK = "_observer_handler"


def _has_marked(logger: logging.Logger, kind: str) -> bool:
    return any(getattr(h, _MARK, None) == kind for h in logger.handlers)


def setup_logging(data_dir: Path | None = None, level: str | None = None) -> None:
    level_name = (level or os.environ.get("OBSERVER_LOG_LEVEL", "INFO")).upper()
    lvl = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(_FORMAT)

    root = logging.getLogger()
    root.setLevel(lvl)

    # Stream to stdout so the journal (and any plain `observer serve`) captures
    # it. Re-add if a later dictConfig removed our handler.
    if not _has_marked(root, "stream"):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        setattr(stream, _MARK, "stream")
        root.addHandler(stream)

    # Also write a rotating file you can tail directly, once a data dir exists.
    if data_dir is not None and not _has_marked(root, "file"):
        try:
            log_dir = Path(data_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            fileh = RotatingFileHandler(
                log_dir / "observer.log",
                maxBytes=5_000_000,
                backupCount=3,
                encoding="utf-8",
            )
            fileh.setFormatter(formatter)
            setattr(fileh, _MARK, "file")
            root.addHandler(fileh)
        except OSError:
            logging.getLogger("observer").warning(
                "could not open log file under %s; logging to stdout only",
                data_dir,
                exc_info=True,
            )

    # Keep our namespace at the chosen level even if a library lowers the root.
    logging.getLogger("observer").setLevel(lvl)
