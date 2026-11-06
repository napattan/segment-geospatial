"""Unit tests for second-pass bug fixes and vectorized geospatial ops."""

import importlib.machinery
import sys
import types
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point, Polygon
import torch

_OPTIONAL_MODS = (
    "segment_anything",
    "segment_anything_hq",
    "rioxarray",
    "xarray",
)
for _mod_name in _OPTIONAL_MODS:
    if _mod_name not in sys.modules:
        try:
            __import__(_mod_name)
        except ImportError:
            _stub = types.ModuleType(_mod_name)
            _stub.__spec__ = importlib.machinery.ModuleSpec(
                _mod_name, loader=None
            )
            _stub.SamAutomaticMaskGenerator = MagicMock()
            _stub.SamPredictor = MagicMock()
            _stub.sam_model_registry = {}
            _stub.DataArray = type("DataArray", (), {})
            sys.modules[_mod_name] = _stub

from samgeo import common  # noqa: E402
from samgeo.hq_sam import SamGeo as HQSamGeo  # noqa: E402
from samgeo.samgeo import SamGeo  # noqa: E402
from samgeo.samgeo2 import SamGeo2  # noqa: E402
from samgeo.samgeo3 import SamGeo3  # noqa: E402


def _create_test_geotiff(path, width=100, height=100, crs="EPSG:32610"):
    transform = from_bounds(
        500000.0, 4100000.0, 501000.0, 4101000.0, width, height
    )
    data = np.zeros((3, height, width), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


def test_tensor_to_numpy_260_masks_no_uint8_overflow():
    """Verify tensor_to_numpy with 260 masks does not wrap mask #256 to 0."""
    h, w = 20, 20
    num_masks = 260
    masks = torch.zeros((num_masks, 3, h, w), dtype=torch.uint8)
    # Mask #256 at i=255: previously (i + 1) == 256 wrapped to 0 in uint8
    masks[255, 1, 15, 15] = 1
    # Overlapping masks i=99 (100) and i=155 (156): 100 + 156 = 256 -> 0
    masks[99, 1, 5, 5] = 1
    masks[155, 1, 5, 5] = 1

    sam = SamGeo.__new__(SamGeo)
    sam.boxes = [[0, 0, 1, 1]] * num_masks
    sam.masks = masks
    sam.image = np.zeros((h, w, 3), dtype=np.uint8)
    sam.source = None

    overlay = sam.tensor_to_numpy(
        index=1, output=None, mask_multiplier=255, dtype="uint8"
    )
    assert overlay.dtype == np.uint8
    assert overlay[15, 15] == 255
    assert overlay[5, 5] == 255
    assert overlay[0, 0] == 0

    sam2 = SamGeo2.__new__(SamGeo2)
    sam2.boxes = [[0, 0, 1, 1]] * num_masks
    sam2.masks = masks.numpy()
    sam2.image = np.zeros((h, w, 3), dtype=np.uint8)
    sam2.source = None

    overlay2 = sam2.tensor_to_numpy(
        index=1, output=None, mask_multiplier=255, dtype="uint8"
    )
    assert overlay2.dtype == np.uint8
    assert overlay2[15, 15] == 255
    assert overlay2[5, 5] == 255
    assert overlay2[0, 0] == 0


def test_coords_to_xy_3d_out_of_bounds_and_bbox_to_xy_batch_parity(tmp_path):
    """Verify 3D coords_to_xy with out-of-bounds points and bbox_to_xy."""
    tif_path = str(tmp_path / "test_utm.tif")
    _create_test_geotiff(tif_path)

    lon, lat = common.transform_coords(
        [500250.0, 500750.0, 600000.0],
        [4100250.0, 4100750.0, 4200000.0],
        "EPSG:32610",
        "EPSG:4326",
    )
    coords_3d = np.array(
        [[[lon[0], lat[0]]], [[lon[1], lat[1]]], [[lon[2], lat[2]]]],
        dtype=float,
    )
    assert coords_3d.shape == (3, 1, 2)

    xy = common.coords_to_xy(tif_path, coords_3d, coord_crs="EPSG:4326")
    assert xy.shape == (2, 1, 2)

    b1_lon, b1_lat = common.transform_coords(
        [500100.0, 500400.0],
        [4100100.0, 4100400.0],
        "EPSG:32610",
        "EPSG:4326",
    )
    b2_lon, b2_lat = common.transform_coords(
        [500500.0, 500800.0],
        [4100500.0, 4100800.0],
        "EPSG:32610",
        "EPSG:4326",
    )
    box1 = [b1_lon[0], b1_lat[0], b1_lon[1], b1_lat[1]]
    box2 = [b2_lon[0], b2_lat[0], b2_lon[1], b2_lat[1]]

    single_1 = common.bbox_to_xy(tif_path, box1, coord_crs="EPSG:4326")
    single_2 = common.bbox_to_xy(tif_path, box2, coord_crs="EPSG:4326")
    batched = common.bbox_to_xy(
        tif_path, [box1, box2], coord_crs="EPSG:4326"
    )

    assert isinstance(single_1[0], int)
    assert batched == [single_1, single_2]


def test_regularize_legacy_and_simplify_preserve_crs():
    """Verify regularize_legacy and simplify preserve GeoDataFrame CRS."""
    poly = Polygon(
        [
            (500000.0, 4100000.0),
            (500100.0, 4100005.0),
            (500100.0, 4100080.0),
            (500000.0, 4100080.0),
        ]
    )
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:32610")

    reg_same_crs = common.regularize_legacy(gdf, crs=None)
    assert reg_same_crs.crs == gdf.crs

    reg_wgs84 = common.regularize_legacy(gdf, crs="EPSG:4326")
    assert reg_wgs84.crs.to_epsg() == 4326

    simplified = common.simplify(gdf, tolerance=1.0)
    assert simplified.crs == gdf.crs

    boxes_gdf = common.boxes_to_vector(
        [[500000.0, 4100000.0, 500100.0, 4100100.0]],
        src_crs="EPSG:32610",
        dst_crs="EPSG:4326",
    )
    assert len(boxes_gdf) == 1
    assert boxes_gdf.crs.to_epsg() == 4326


def test_region_groups_lut_and_single_pixel_minor_axis_zero():
    """Verify region_groups LUT filtering and axis_minor_length == 0 guard."""
    img = np.zeros((30, 30), dtype=np.uint8)
    img[2, 2] = 1
    img[5:8, 5:8] = 1
    img[15:25, 15:25] = 1

    labeled, df = common.region_groups(img, min_size=5, max_size=50)
    assert len(df) == 1
    assert df["area"].iloc[0] == 9
    assert int(labeled.max()) == 1

    labeled_all, df_all = common.region_groups(img, min_size=1)
    assert len(df_all) == 3
    single_px_row = df_all[df_all["area"] == 1].iloc[0]
    assert single_px_row["axis_minor_length"] == 0
    assert np.isnan(single_px_row["elongation"])
    multi_px_rows = df_all[df_all["area"] > 1]
    assert np.all(np.isfinite(multi_px_rows["elongation"]))


def test_predict_labels_arg_and_batched_box_transforms(tmp_path):
    """Verify predict_torch labels, SamGeo2 state reset, and SamGeo3 batch."""
    for cls in (SamGeo, HQSamGeo):
        inst = cls.__new__(cls)
        inst.image = np.zeros((32, 32, 3), dtype=np.uint8)
        inst.device = "cpu"
        inst.predictor = MagicMock()
        inst.predictor.transform.apply_boxes_torch.side_effect = (
            lambda b, s: b
        )
        inst.predictor.predict_torch.return_value = (
            "masks",
            "scores",
            "logits",
        )

        pts = np.array([[10, 10], [20, 20]])
        lbls = np.array([1, 0])
        boxes = [[0, 0, 5, 5], [10, 10, 15, 15]]
        inst.predict(
            point_coords=pts,
            point_labels=lbls,
            boxes=boxes,
            return_results=True,
        )
        call_kwargs = inst.predictor.predict_torch.call_args.kwargs
        assert np.array_equal(call_kwargs["point_labels"], lbls)

    sam2 = SamGeo2.__new__(SamGeo2)
    sam2.predictor = MagicMock()
    sam2.image_batch = ["stale_batch"]
    sam2.set_image(np.zeros((16, 16, 3), dtype=np.uint8))
    assert sam2.image_batch is None

    tif_path = str(tmp_path / "sam3_boxes.tif")
    _create_test_geotiff(tif_path)

    sam2.source = tif_path
    sam2.predictor.predict_batch.return_value = (
        [np.ones((2, 16, 16), dtype=bool)],
        [np.array([0.9, 0.8])],
        [np.zeros((2, 16, 16), dtype=np.float32)],
    )
    lon, lat = common.transform_coords(
        [500200.0, 500600.0],
        [4100200.0, 4100600.0],
        "EPSG:32610",
        "EPSG:4326",
    )
    pts_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon[0], lat[0]), Point(lon[1], lat[1])],
        crs="EPSG:4326",
    )
    sam2.predict_by_points(point_coords_batch=pts_gdf, point_crs=None)
    assert len(sam2.masks) == 2

    sam3 = SamGeo3.__new__(SamGeo3)
    sam3.backend = "meta"
    sam3.source = tif_path
    sam3.image = np.zeros((100, 100, 3), dtype=np.uint8)
    sam3.inference_state = {"state": True}
    sam3.model = MagicMock()
    sam3.model.inst_interactive_predictor = True
    sam3.model.predict_inst.return_value = (
        np.ones((2, 100, 100), dtype=bool),
        np.array([0.9, 0.8]),
        np.zeros((2, 100, 100), dtype=np.float32),
    )
    sam3._meta_autocast = MagicMock()
    sam3._meta_autocast.return_value.__enter__.return_value = None
    sam3._meta_autocast.return_value.__exit__.return_value = None

    boxes_wgs84 = [
        [lon[0], lat[0], lon[1], lat[1]],
        [lon[0], lat[0], lon[1], lat[1]],
    ]
    with patch(
        "samgeo.common.coords_to_xy", wraps=common.coords_to_xy
    ) as spy_xy:
        sam3.predict_inst(box=boxes_wgs84, box_crs="EPSG:4326")
        assert spy_xy.call_count == 1
