"""Optional Hailo detector backend for Raspberry Pi (Hailo-8/8L accelerator).

This runs a YOLO model precompiled to a ``.hef`` via the Hailo Dataflow Compiler
using the HailoRT Python API. ``hailort`` is only available on the device, so it
is imported lazily and this backend is only constructed when
``settings.detector_backend == "hailo"``.

NOTE: HEF input/output tensor names and the exact post-processing depend on how
the model was compiled. The decode below targets a standard YOLOv8 export; adjust
``_postprocess`` to match your compiled model. Verify on-device (milestone 5).
"""

from __future__ import annotations

import numpy as np

from observer.config import Settings
from observer.pipeline.detector.base import Detection


class HailoDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._infer = None
        self._input_shape: tuple[int, int] | None = None

    def _ensure_model(self):
        if self._infer is not None:
            return
        # Imported here so the package imports cleanly off-device.
        from hailo_platform import (  # type: ignore
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
        configure_params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        network_group = self._vdevice.configure(hef, configure_params)[0]
        self._network_group = network_group
        in_params = InputVStreamParams.make(network_group)
        out_params = OutputVStreamParams.make(network_group)
        in_info = hef.get_input_vstream_infos()[0]
        self._input_shape = (in_info.shape[0], in_info.shape[1])  # (h, w)
        self._infer = InferVStreams(network_group, in_params, out_params)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._ensure_model()
        assert self._input_shape is not None
        import cv2

        h, w = self._input_shape
        resized = cv2.resize(frame, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(rgb, axis=0).astype(np.float32)

        with self._network_group.activate():
            with self._infer as pipeline:
                raw = pipeline.infer(batch)
        return self._postprocess(raw, frame.shape[1], frame.shape[0])

    def _postprocess(
        self, raw: dict, orig_w: int, orig_h: int
    ) -> list[Detection]:
        """Decode HEF output into detections.

        Placeholder for the standard Hailo YOLOv8 NMS output format
        (per-class lists of [y_min, x_min, y_max, x_max, score]). Wire this to
        your compiled model's actual output layer name and layout on-device.
        """
        out: list[Detection] = []
        for class_id, dets in enumerate(next(iter(raw.values()))):
            if dets is None or len(dets) == 0:
                continue
            for det in dets:
                y_min, x_min, y_max, x_max, score = det[:5]
                if score < self._settings.detect_conf:
                    continue
                out.append(
                    Detection(
                        xyxy=(
                            x_min * orig_w,
                            y_min * orig_h,
                            x_max * orig_w,
                            y_max * orig_h,
                        ),
                        class_id=int(class_id),
                        confidence=float(score),
                    )
                )
        return out
