from ais_validation import clean_report


def test_valid_report_passes_through():
    report, reason = clean_report(366123456, 51.5, 1.2, 12.3, 90.0)
    assert reason is None
    assert (report.lat, report.lon, report.speed, report.course) == (51.5, 1.2, 12.3, 90.0)


def test_speed_not_available_becomes_none_not_a_spike():
    for sog in (102.2, 102.3):
        report, reason = clean_report(366123456, 51.5, 1.2, sog)
        assert reason is None and report.speed is None


def test_course_not_available_becomes_none():
    report, _ = clean_report(366123456, 51.5, 1.2, 5.0, 360.0)
    assert report.course is None


def test_position_not_available_is_rejected():
    assert clean_report(366123456, 91.0, 1.2, 5.0)[1] == "position_not_available"
    assert clean_report(366123456, 51.0, 181.0, 5.0)[1] == "position_not_available"


def test_null_island_is_rejected():
    assert clean_report(366123456, 0.0, 0.0, 5.0)[1] == "null_island"


def test_placeholder_and_non_vessel_mmsis_are_rejected():
    for mmsi in (999999999, 123456789, 2190047, 970123456, 111232511):
        assert clean_report(mmsi, 51.0, 1.0, 5.0)[1] == "non_vessel_or_placeholder_mmsi"
