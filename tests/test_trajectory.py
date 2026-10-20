from types import SimpleNamespace

from trajectory import FEATURE_NAMES, angle_diff, trajectory_features


def rep(lat, lon, speed=10.0):
    return SimpleNamespace(lat=lat, lon=lon, speed=speed, course=None)


def feats(points):
    return dict(zip(FEATURE_NAMES, trajectory_features([rep(*p) for p in points])))


def test_straight_track_is_straight_with_no_turning():
    f = feats([(10.0, 60.0 + 0.01 * i) for i in range(8)])
    assert f["straightness"] > 0.99
    assert f["mean_turn_deg"] < 1.0


def test_zigzag_at_constant_speed_is_detected_by_turning_not_speed():
    zig = [(10.0 + (0.01 if i % 2 else 0.0), 60.0 + 0.01 * i) for i in range(8)]
    f = feats(zig)
    assert f["speed_std"] == 0.0           # same speed throughout
    assert f["mean_turn_deg"] > 45.0       # but heading keeps flipping
    assert f["straightness"] < 0.8


def test_sudden_stop_shows_as_max_speed_change():
    pts = [(10.0, 60.0 + 0.01 * i, 14.0) for i in range(5)] + [(10.0, 60.05, 0.5)]
    assert feats(pts)["max_speed_change"] == 13.5


def test_stationary_ship_is_not_called_erratic():
    jitter = [(10.0 + 1e-5 * (i % 2), 60.0, 0.1) for i in range(8)]  # ~1 m GPS noise
    f = feats(jitter)
    assert f["mean_turn_deg"] == 0.0 and f["straightness"] == 1.0


def test_unknown_speeds_are_ignored():
    f = feats([(10.0, 60.0, None), (10.0, 60.01, 10.0), (10.0, 60.02, None), (10.0, 60.03, 12.0)])
    assert f["mean_speed"] == 11.0 and f["max_speed_change"] == 0.0


def test_angle_diff_wraps():
    assert angle_diff(350, 10) == 20
    assert angle_diff(10, 190) == 180


def test_features_contain_no_position():
    assert not any("lat" in n or "lon" in n for n in FEATURE_NAMES)
