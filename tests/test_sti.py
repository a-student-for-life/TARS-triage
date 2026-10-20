"""STI preprocessing: reproducibility and behaviour (no TensorFlow needed)."""
import cv2
import numpy as np
import pytest

import ml_logic


@pytest.fixture
def preview(tmp_path):
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(50, 15, (400, 400)), 0, 255).astype(np.uint8)
    img[150:250, 100:300] = 5  # dark slick-like patch
    path = tmp_path / "S1A_IW_GRDH_TEST.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 30])
    return str(path)


def test_same_image_gives_identical_draws(preview):
    a = ml_logic.sti_batch(preview, n_draws=4, sigma=0.5)
    b = ml_logic.sti_batch(preview, n_draws=4, sigma=0.5)
    assert a.shape == (4, 380, 380, 3)
    assert np.array_equal(a, b)


def test_draws_differ_from_each_other(preview):
    batch = ml_logic.sti_batch(preview, n_draws=3, sigma=0.5)
    assert not np.array_equal(batch[0], batch[1])


def test_sigma_zero_is_deterministic_single_draw(preview):
    batch = ml_logic.sti_batch(preview, n_draws=8, sigma=0.0)
    assert batch.shape[0] == 1


def test_dark_patch_stays_darker_than_sea(preview):
    out = ml_logic.preprocess_style(preview, sigma=0.5, rng=np.random.default_rng(1))[:, :, 0]
    patch = out[150 * 380 // 400:250 * 380 // 400, 100 * 380 // 400:300 * 380 // 400]
    assert patch.mean() < out.mean()


def test_brightness_is_shifted_into_training_range(preview):
    out = ml_logic.preprocess_style(preview, sigma=0.0)
    assert out.mean() > 100  # raw preview mean is ~50


def test_no_debug_file_written_by_default(preview, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ml_logic.preprocess_style(preview)
    assert not (tmp_path / "debug_ai_input_v3.jpg").exists()


def test_global_rng_path_still_patchable_for_old_scripts(preview, monkeypatch):
    """scripts/noise_sensitivity.py patches numpy.random.normal; keep that working."""
    calls = []
    real = np.random.normal

    def spy(loc, scale, size):
        calls.append(scale)
        return real(loc, scale, size)

    monkeypatch.setattr(np.random, "normal", spy)
    ml_logic.preprocess_style(preview)
    assert calls == [0.5]


def test_backbone_is_built_with_imagenet_flag(monkeypatch):
    """Regression: Keras's EfficientNet adds an input Rescaling layer only when
    weights="imagenet". Building with weights=None changed every score
    (corpus AUC 0.9904 -> 0.9641) even though the trained weights loaded fine."""
    pytest.importorskip("tensorflow")
    from tensorflow.keras import applications

    seen = {}

    def fake_b4(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after capturing arguments")

    monkeypatch.setattr(applications, "EfficientNetB4", fake_b4)
    with pytest.raises(RuntimeError):
        ml_logic.build_model_architecture()
    assert seen["weights"] == "imagenet"


@pytest.fixture
def regional(monkeypatch):
    import os
    import config
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr(config, "REGIONAL_HEADS", os.path.join(repo, "models", "regional_heads.json"))
    monkeypatch.setattr(config, "MODEL_KIND", "regional")
    monkeypatch.setattr(ml_logic, "_registry", None)
    return config


def test_vessel_location_selects_the_regional_head(regional):
    assert ml_logic.choose_head(33.0, 32.0) == "eastern_med_v1"     # off Israel/Lebanon
    assert ml_logic.choose_head(26.5, 52.0) == "arabian_gulf_v1"    # central Gulf
    assert ml_logic.choose_head(45.0, -30.0) == "csiro_raw"         # mid-Atlantic: fallback
    assert ml_logic.choose_head(None, None) == "csiro_raw"


def test_browse_v1_mode_forces_the_mediterranean_head(regional, monkeypatch):
    monkeypatch.setattr(regional, "MODEL_KIND", "browse_v1")
    assert ml_logic.choose_head(26.5, 52.0) == "eastern_med_v1"


def test_regional_analysis_uses_raw_crop_and_that_heads_threshold(regional, preview, monkeypatch):
    seen = {}

    class FakeHead:
        def predict(self, feats, verbose=0):
            seen["feats"] = feats.shape
            return np.array([[0.5]])

    def fake_features(batch):
        seen["batch"] = batch.shape
        return np.zeros((batch.shape[0], 1792))

    monkeypatch.setattr(ml_logic, "_features", fake_features)
    monkeypatch.setattr(ml_logic, "_head", lambda name: (FakeHead(), 0.46 if name == "arabian_gulf_v1" else 0.742))
    gulf = ml_logic.analyze_image(preview, lat=26.5, lon=52.0)
    med = ml_logic.analyze_image(preview, lat=33.0, lon=32.0)
    assert seen["batch"] == (1, 380, 380, 3)                          # one raw pass, no STI draws
    assert gulf["model"] == "arabian_gulf_v1" and gulf["is_oil"] and gulf["threshold_used"] == 0.46
    assert med["model"] == "eastern_med_v1" and not med["is_oil"]


def test_registry_files_exist_and_thresholds_are_valid(regional):
    import json
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for h in ml_logic.registry()["heads"]:
        assert os.path.exists(os.path.join(repo, h["weights"]))
        th = json.load(open(os.path.join(repo, h["info"])))["threshold"]
        assert 0 < th < 1
