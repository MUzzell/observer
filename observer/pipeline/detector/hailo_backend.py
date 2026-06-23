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

import numpy as np

from observer.config import Settings
from observer.pipeline.detector.base import Detection


class HailoDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ready = False

    def _ensure_model(self) -> None:
        if self._ready:
            return
        from hailo_platform import (  # type: ignore  # provided by HailoRT on the Pi
            HEF,
            ConfigureParams,
            HailoStreamInterface,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        hef = HEF(str(self._settings.hailo_hef_path))
        self._vdevice = VDevice()
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        self._network_group = self._vdevice.configure(hef, cfg)[0]
        self._in_params = InputVStreamParams.make(self._network_group)
        self._out_params = OutputVStreamParams.make(self._network_group)
        in_info = hef.get_input_vstream_infos()[0]
        self._input_name = in_info.name
        # in_info.shape is (height, width, channels)
        self._in_h, self._in_w = in_info.shape[0], in_info.shape[1]
        self._InferVStreams = InferVStreams
        self._ready = True

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._ensure_model()
        import cv2

        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, (self._in_w, self._in_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(rgb, 0).astype(np.float32)

        with self._network_group.activate():
            with self._InferVStreams(
                self._network_group, self._in_params, self._out_params
            ) as pipeline:
                raw = pipeline.infer({self._input_name: batch})
        return self._postprocess(raw, orig_w, orig_h)

    def _postprocess(self, raw: dict, orig_w: int, orig_h: int) -> list[Detection]:
        # NMS-on-chip output: {output_name: [ per-class array of [y0,x0,y1,x1,score] ]}
        detections = next(iter(raw.values()))
        # Batch dimension if present.
        if isinstance(detections, np.ndarray) and detections.ndim and len(detections):
            detections = detections[0]
        out: list[Detection] = []
        for class_dets in detections:  # one entry per class; we have a single class
            if class_dets is None or len(class_dets) == 0:
                continue
            for det in class_dets:
                y0, x0, y1, x1, score = det[:5]
                if score < self._settings.detect_conf:
                    continue
                out.append(
                    Detection(
                        xyxy=(x0 * orig_w, y0 * orig_h, x1 * orig_w, y1 * orig_h),
                        label="aircraft",
                        confidence=float(score),
                    )
                )
        return out

    def classify_type(self, frame: np.ndarray) -> tuple[str | None, float]:
        return None, 0.0
