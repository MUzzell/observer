"""Unit tests for capture-timestamp parsing."""

from __future__ import annotations

from datetime import datetime

from observer.naming import parse_capture_time


def test_parses_camera_stamp():
    assert parse_capture_time("0-141-20260327132601.mp4") == datetime(2026, 3, 27, 13, 26, 1)


def test_parses_second_sample():
    assert parse_capture_time("0-956-20260509172634.mp4") == datetime(2026, 5, 9, 17, 26, 34)


def test_uuid_filename_has_no_timestamp():
    assert parse_capture_time("fcd125a7-e450-4c3e-8d60-15ee9395c233.mp4") is None


def test_invalid_datetime_digits_rejected():
    # 14 digits but not a valid date (month 13) -> None, not a crash.
    assert parse_capture_time("clip-20261345996000.mp4") is None
