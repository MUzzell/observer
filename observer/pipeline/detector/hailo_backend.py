"""Hailo detector backend for the processor RPi (Hailo-8/8L).

Runs the fine-tuned YOLOv8n aircraft model compiled to a ``.hef`` via the Hailo
Dataflow Compiler. ``hailort`` only exists on the device, so it is imported
lazily and this backend is constructed only when
``settings.detector_backend == "hailo"``.

IMPORTANT — this cannot be validated off-device. The output parsing below assumes
the HEF was compiled **with NMS on-chip** (the recommended Hailo YOLOv8 path),
whose output is, per class, an array of ``[y0, x0, y1, x1, score]`` with
normalized coordinates. If you compile without NMS you must decode raw tensors
here instead. Verify the actual output names/shape on the Pi with
``hailortcli parse-hef <model>.hef`` and adjust ``_postprocess`` to match.

Single class only (class 0 = aircraft); ``classify_type`` is unsupported.
"""

from __future__ import annotations

import logging

import numpy as np

from observer.config import Settings
from observer.pipeline.detector.base import Detection

log = logging.getLogger("observer.detector.hailo")


class HailoDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ready = False
        self._logged_shape = False

    def _ensure_model(self) -> None:
        if self._ready:
            return
        hef_path = str(self._settings.hailo_hef_path)
        log.info("initialising Hailo backend (hef=%s)", hef_path)
        try:
            from hailo_platform import (  # type: ignore  # provided by HailoRT on the Pi
                HEF,
                ConfigureParams,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
        except Exception:
            log.exception(
                "failed to import hailo_platform — is HailoRT installed and "
                "exposed to this venv? (see docs/processor-pi-install.md §7c)"
            )
            raise

        if not self._settings.hailo_hef_path.exists():
            log.error("Hailo HEF not found at %s", hef_path)
            raise FileNotFoundError(f"Hailo HEF not found: {hef_path}")

        try:
            hef = HEF(hef_path)
            self._vdevice = VDevice()
            cfg = ConfigureParams.create_from_hef(
                hef, interface=HailoStreamInterface.PCIe
            )
            self._network_group = self._vdevice.configure(hef, cfg)[0]
            self._in_params = InputVStreamParams.make(self._network_group)
            self._out_params = OutputVStreamParams.make(self._network_group)
            in_info = hef.get_input_vstream_infos()[0]
            self._input_name = in_info.name
            # in_info.shape is (height, width, channels)
            self._in_h, self._in_w = in_info.shape[0], in_info.shape[1]
            self._InferVStreams = InferVStreams
        except Exception:
            log.exception("failed to configure Hailo device from %s", hef_path)
            raise

        self._ready = True
        log.info(
            "Hailo backend ready (input=%r, %dx%d)",
            self._input_name, self._in_w, self._in_h,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._ensure_model()
        import cv2

        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, (self._in_w, self._in_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # The HEF's input stream is uint8 (normalisation happens on-chip); feed
        # uint8 directly to avoid a per-frame float32->uint8 conversion.
        batch = np.expand_dims(rgb, 0).astype(np.uint8)

        with self._network_group.activate():
            with self._InferVStreams(
                self._network_group, self._in_params, self._out_params
            ) as pipeline:
                raw = pipeline.infer({self._input_name: batch})
        dets = self._postprocess(raw, orig_w, orig_h)
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "hailo infer: %d detection(s) above conf=%.2f",
                len(dets), self._settings.detect_conf,
            )
        return dets

    def _postprocess(self, raw: dict, orig_w: int, orig_h: int) -> list[Detection]:
        # NMS-on-chip output: per-class arrays of [y0, x0, y1, x1, score] rows,
        # but the exact nesting (batch / class wrapping, object vs float arrays)
        # varies by HailoRT version and HEF. Log the real structure once so it's
        # verifiable from the journal, then walk it depth-first collecting any
        # 5-wide rows — robust to the wrapping, and single-class here so we don't
        # need to preserve which class a row came from.
        if not self._logged_shape:
            self._logged_shape = True
            log.info(
                "hailo raw output structure: %s",
                {k: _describe(v) for k, v in raw.items()},
            )

        detections = next(iter(raw.values()))
        out: list[Detection] = []
        for box in _iter_boxes(detections):
            if len(box) < 5:
                continue
            y0, x0, y1, x1, score = (float(v) for v in box[:5])
            if score < self._settings.detect_conf:
                continue
            out.append(
                Detection(
                    xyxy=(x0 * orig_w, y0 * orig_h, x1 * orig_w, y1 * orig_h),
                    label="aircraft",
                    confidence=score,
                )
            )
        return out

    def classify_type(self, frame: np.ndarray) -> tuple[str | None, float]:
        return None, 0.0


_NUMERIC = (int, float, np.floating, np.integer)


def _iter_boxes(node):
    """Yield every ``[y0, x0, y1, x1, score, ...]`` detection row found anywhere
    in the (variably nested) NMS output, skipping empty class entries."""
    if isinstance(node, np.ndarray):
        if node.dtype == object:
            for el in node:
                yield from _iter_boxes(el)
        elif node.ndim >= 2 and node.shape[-1] >= 5:
            for row in node.reshape(-1, node.shape[-1]):
                yield row
        elif node.ndim == 1 and node.shape[0] >= 5:
            yield node
        elif node.ndim >= 1:
            for el in node:
                yield from _iter_boxes(el)
        return
    if isinstance(node, (list, tuple)):
        if len(node) >= 5 and all(isinstance(v, _NUMERIC) for v in node[:5]):
            yield node
        else:
            for el in node:
                yield from _iter_boxes(el)


def _describe(node, depth: int = 0) -> str:
    """Compact, recursive description of an inference-output node for logging."""
    if depth > 4:
        return "..."
    if isinstance(node, np.ndarray):
        return f"ndarray(shape={node.shape}, dtype={node.dtype})"
    if isinstance(node, (list, tuple)):
        head = _describe(node[0], depth + 1) if node else "empty"
        return f"{type(node).__name__}(len={len(node)}, [0]={head})"
    return type(node).__name__
