import numpy as np

from browse_geo import crop_around, latlon_to_pixel, pixel_size_km

GEOM = {"type": "Polygon", "coordinates": [[[30, 31], [33, 31], [33, 34], [30, 34], [30, 31]]]}


def test_north_up_bbox_mapping():
    assert latlon_to_pixel(GEOM, 34, 30, 900, 1200) == (0, 0)              # NW corner -> top-left
    assert latlon_to_pixel(GEOM, 31, 33, 900, 1200) == (1200, 900)         # SE corner -> bottom-right
    x, y = latlon_to_pixel(GEOM, 32.5, 31.5, 900, 1200)
    assert (x, y) == (600, 450)


def test_crop_is_centred_and_sized_in_km():
    img = np.full((900, 1200), 80, np.uint8)
    img[440:461, 590:611] = 255                                             # marker at the centre
    crop, info = crop_around(img, GEOM, 32.5, 31.5, crop_km=25)
    assert info["reason"] is None
    side_km = crop.shape[1] * pixel_size_km(GEOM, 1200)
    assert abs(side_km - 25) < 1.0
    cy, cx = np.argwhere(crop == 255).mean(axis=0)
    assert abs(cx - crop.shape[1] / 2) < 2 and abs(cy - crop.shape[0] / 2) < 2


def test_vessel_at_edge_is_rejected():
    img = np.full((900, 1200), 80, np.uint8)
    crop, info = crop_around(img, GEOM, 33.99, 30.01, crop_km=25)
    assert crop is None and info["reason"] == "vessel_too_close_to_image_edge"


def test_mostly_no_data_is_rejected():
    img = np.zeros((900, 1200), np.uint8)                                    # outside the footprint
    crop, info = crop_around(img, GEOM, 32.5, 31.5, crop_km=25)
    assert crop is None and info["reason"] == "vessel_outside_imaged_area"
