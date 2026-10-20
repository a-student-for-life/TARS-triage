from coast import near_land


def test_mid_ocean_is_offshore():
    assert not near_land([-45.0, 40.0], [-120.0, -40.0], 10.0).any()


def test_harbour_and_land_are_near_land():
    assert near_land([43.296, 51.5], [5.364, -0.12], 10.0).all()


def test_point_just_off_a_coast_is_near_land_at_larger_radius():
    # ~15 km west of the Portuguese coast near Lisbon
    assert near_land([38.72], [-9.68], 30.0)[0]
