import numpy as np

import cv2

from browse_geo import crop_around, is_map_projected, latlon_to_pixel, pixel_size_km

GEOM = {"type": "Polygon", "coordinates": [[[30, 31], [33, 31], [33, 34], [30, 34], [30, 31]]]}


def projected_image(h=900, w=1200, value=80):
    """Synthetic map-projected preview: data inside a tilted footprint, black corners."""
    img = np.zeros((h, w), np.uint8)
    poly = np.array([[int(0.08 * w), 0], [w, int(0.06 * h)], [int(0.92 * w), h], [0, int(0.94 * h)]], np.int32)
    cv2.fillPoly(img, [poly], value)
    return img


def test_north_up_bbox_mapping():
    assert latlon_to_pixel(GEOM, 34, 30, 900, 1200) == (0, 0)              # NW corner -> top-left
    assert latlon_to_pixel(GEOM, 31, 33, 900, 1200) == (1200, 900)         # SE corner -> bottom-right
    x, y = latlon_to_pixel(GEOM, 32.5, 31.5, 900, 1200)
    assert (x, y) == (600, 450)


def test_crop_is_centred_and_sized_in_km():
    img = projected_image()
    img[440:461, 590:611] = 255                                             # marker at the centre
    crop, info = crop_around(img, GEOM, 32.5, 31.5, crop_km=25)
    assert info["reason"] is None
    side_km = crop.shape[1] * pixel_size_km(GEOM, 1200)
    assert abs(side_km - 25) < 1.0
    cy, cx = np.argwhere(crop == 255).mean(axis=0)
    assert abs(cx - crop.shape[1] / 2) < 2 and abs(cy - crop.shape[0] / 2) < 2


def test_vessel_at_edge_is_rejected():
    img = projected_image()
    crop, info = crop_around(img, GEOM, 33.99, 30.01, crop_km=25)
    assert crop is None and info["reason"] == "vessel_too_close_to_image_edge"


def test_mostly_no_data_is_rejected():
    img = np.zeros((900, 1200), np.uint8)                                    # outside the footprint
    crop, info = crop_around(img, GEOM, 32.5, 31.5, crop_km=25)
    assert crop is None and info["reason"] == "vessel_outside_imaged_area"


# A real Sentinel-1 footprint in ASF corner order (Gulf of Oman scene, radar-geometry preview)
RADAR_GEOM = {"type": "Polygon", "coordinates": [[[57.99934, 24.946878], [58.324478, 23.438196], [60.811909, 23.86442],
                                                  [60.517208, 25.370054], [57.99934, 24.946878]]]}


def test_image_type_detection():
    assert is_map_projected(projected_image())
    assert not is_map_projected(np.full((663, 1024), 80, np.uint8))


def test_radar_geometry_corners_map_to_image_corners():
    h, w = 663, 1024
    pts = RADAR_GEOM["coordinates"][0]
    for (lon, lat), (ex, ey) in zip([pts[1], pts[2], pts[3], pts[0]], [(0, 0), (w, 0), (w, h), (0, h)]):
        x, y = latlon_to_pixel(RADAR_GEOM, lat, lon, h, w, projected=False)
        assert abs(x - ex) < 0.5 and abs(y - ey) < 0.5


def test_radar_geometry_crop_is_centred_on_the_vessel():
    h, w = 663, 1024
    img = np.full((h, w), 80, np.uint8)                                      # full frame: radar geometry
    lat, lon = 23.8, 58.7                                                   # ~105 km from where the old rule put it
    x, y = latlon_to_pixel(RADAR_GEOM, lat, lon, h, w, projected=False)
    img[int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4] = 255
    crop, info = crop_around(img, RADAR_GEOM, lat, lon, crop_km=25)
    assert info["reason"] is None and info["geometry"] == "radar"
    cy, cx = np.argwhere(crop == 255).mean(axis=0)
    assert abs(cx - crop.shape[1] / 2) < 2 and abs(cy - crop.shape[0] / 2) < 2
    side_km = crop.shape[1] * pixel_size_km(RADAR_GEOM, w, projected=False)
    assert abs(side_km - 25) < 1.0
    # the old bounding-box assumption would have cut a window far from the vessel
    xb, yb = latlon_to_pixel(RADAR_GEOM, lat, lon, h, w, projected=True)
    assert np.hypot(xb - x, yb - y) * pixel_size_km(RADAR_GEOM, w, projected=False) > 20
