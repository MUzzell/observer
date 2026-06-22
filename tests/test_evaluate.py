"""Unit tests for the evaluator's scoring (pure, no model/video)."""

from __future__ import annotations

from observer.evaluate import Sample, _predict, metrics_at


def test_predict_persistence_and_strong_rules():
    # 3 hits at >=0.30 -> present
    assert _predict([0.31, 0.4, 0.5], present=0.30, hits=3, strong=0.55) is True
    # only 1 hit, but a strong one -> present via strong rule
    assert _predict([0.1, 0.62], present=0.30, hits=3, strong=0.55) is True
    # a lone weak blip -> absent
    assert _predict([0.16], present=0.30, hits=3, strong=0.55) is False


def test_metrics_confusion_and_scores():
    samples = [
        Sample("h1", True, [0.6, 0.7, 0.65]),   # aircraft, predicted yes -> TP
        Sample("h2", True, [0.1]),               # aircraft, predicted no  -> FN
        Sample("b1", False, [0.16]),             # none, predicted no      -> TN
        Sample("b2", False, [0.4, 0.5, 0.45]),   # none, predicted yes     -> FP
    ]
    m = metrics_at(samples, present=0.30, hits=3, strong=0.55)
    assert (m.tp, m.fn, m.tn, m.fp) == (1, 1, 1, 1)
    assert m.precision == 0.5 and m.recall == 0.5
    assert abs(m.f1 - 0.5) < 1e-9
    assert m.accuracy == 0.5
