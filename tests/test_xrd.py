import io

import h5py
import numpy as np
import pandas as pd
import pytest

from backend import xrd

H, W, N = 40, 50, 6
SPOT = (5, 30, 3)  # (row, col, frame number)
CENTER = (20.0, 10.0)  # beam center (x, y) in pixels


def make_dataset(folder, with_data=True):
    """Eiger-style master + external data file with Poisson background, one spot, a mask and an overflow."""
    rng = np.random.default_rng(0)
    frames = rng.poisson(100, size=(N, H, W)).astype(np.uint32)
    frames[SPOT[2] - 1, SPOT[0], SPOT[1]] += 400
    frames[:, :, 0] = np.uint32(0xFFFFFFFF)  # masked column
    frames[2, 30, 40] = 4_000_000  # overflow
    mask = np.zeros((H, W), np.uint32)
    mask[:, 0] = 1

    if with_data:
        with h5py.File(folder / "t_data_000001.h5", "w") as f:
            f["entry/data/data"] = frames
    master = folder / "t_master.h5"
    with h5py.File(master, "w") as f:
        f["entry/data/data_000001"] = h5py.ExternalLink("t_data_000001.h5", "/entry/data/data")
        det = "entry/instrument/detector"
        f[f"{det}/detector_distance"] = np.float32(0.1)
        f[f"{det}/beam_center_x"] = np.float32(CENTER[0])
        f[f"{det}/beam_center_y"] = np.float32(CENTER[1])
        f[f"{det}/x_pixel_size"] = np.float32(75e-6)
        f[f"{det}/y_pixel_size"] = np.float32(75e-6)
        f[f"{det}/detectorSpecific/pixel_mask"] = mask
        f[f"{det}/detectorSpecific/countrate_correction_count_cutoff"] = np.uint32(3_579_750)
        f["entry/instrument/beam/incident_wavelength"] = np.float32(0.6)
    return master


@pytest.fixture
def dataset(tmp_path):
    master = make_dataset(tmp_path)
    source = xrd.FrameSource(master)
    header = xrd.read_header(master, source)
    stats = xrd.compute_stats(source, header, xrd.read_pixel_mask(master, (H, W)))
    return source, header, stats


def test_header_and_stats(dataset):
    source, header, stats = dataset
    assert (header.n_frames, header.height, header.width) == (N, H, W)
    assert stats.n_masked == H and stats.n_overflow == 1
    assert not stats.valid[:, 0].any() and not stats.valid[30, 40]
    assert stats.scale == pytest.approx(1.0, abs=0.01)


def test_missing_data_file(tmp_path):
    source = xrd.FrameSource(make_dataset(tmp_path, with_data=False))
    assert source.missing_files == ["t_data_000001.h5"]
    with pytest.raises(FileNotFoundError):
        next(source.iter_frames())


def test_z_map_spot_and_invalid(dataset):
    source, _, stats = dataset
    z = xrd.intensity_map(source, stats, [SPOT[2]])
    assert z[SPOT[0], SPOT[1]] > 20
    assert np.nanmin(z) >= 0
    assert np.isnan(z[:, 0]).all() and np.isnan(z[30, 40])
    assert np.nanmax(np.delete(z.ravel(), SPOT[0] * W + SPOT[1])) < 8  # rest is noise
    # Spot is only in its own frame; a range keeps it through the per-pixel max.
    assert xrd.intensity_map(source, stats, [1])[SPOT[0], SPOT[1]] < 8
    assert xrd.intensity_map(source, stats, list(range(1, N + 1)))[SPOT[0], SPOT[1]] > 20


def test_counts_map(dataset):
    source, _, stats = dataset
    counts = xrd.intensity_map(source, stats, [1, 2], mode="counts")
    raw = np.stack([f for _, f in source.iter_frames([1, 2])]).astype(float).sum(0)
    assert counts[10, 10] == raw[10, 10]


def test_q_orientation(dataset):
    _, header, _ = dataset
    qx, qy = xrd.q_components(xrd.header_integrator(header), (H, W))
    cx, cy = CENTER
    assert qy[0, int(cx)] > 0 > qy[H - 1, int(cx)]  # image top -> +qy
    assert qx[int(cy), W - 1] > 0 > qx[int(cy), 0]  # image right -> +qx
    assert abs(qx[int(cy), int(cx)]) < 0.01 and abs(qy[int(cy), int(cx)]) < 0.01


def test_bin_to_q_layout():
    values = np.array([[1.0, 2.0], [3.0, np.nan]])
    qx = np.array([[0.001, 0.005], [0.001, 0.005]])
    qy = np.array([[0.005, 0.005], [0.001, 0.001]])
    grid = xrd.bin_to_q(values, qx, qy, 0.004)
    assert grid.qx == pytest.approx([0.002, 0.006])
    assert grid.qy == pytest.approx([0.006, 0.002])  # descending: top row = largest qy
    np.testing.assert_array_equal(grid.values, [[1.0, 2.0], [3.0, np.nan]])


def test_bin_to_q_max_and_mean():
    values = np.array([1.0, 5.0, 3.0])
    q = np.array([0.001, 0.002, 0.003])
    assert xrd.bin_to_q(values, q, q, 0.004).values[0, 0] == 5.0
    assert xrd.bin_to_q(values, q, q, 0.004, agg="mean").values[0, 0] == 3.0


def test_parse_ranges():
    r = xrd.parse_ranges("19; 9-29，1-150", 150)
    assert [x.label for x in r] == ["19", "9-29", "1-150"]
    assert xrd.parse_ranges("5-3", 10)[0].frames == [3, 4, 5]
    for bad in ["0", "1-151", "abc", ""]:
        with pytest.raises(ValueError):
            xrd.parse_ranges(bad, 150)


VALUES = np.array([[0.04, np.nan, 12.345], [1.0, 2.0, 3.0]])
GRID = xrd.QGrid(np.array([0.37, 0.374]), np.array([0.922, 0.918]), np.array([[1.0, np.nan], [2.0, 3.0]]))


def test_matrix_and_xyz_csv():
    tables = xrd.pixel_tables(VALUES, ["matrix", "xyz"])
    pixel = xrd.table_csv(tables["pixel"])
    matrix = pd.read_csv(io.BytesIO(pixel), index_col=0)
    assert list(matrix.columns) == ["0", "1", "2"] and list(matrix.index) == [0, 1]
    assert np.isnan(matrix.iloc[0, 1]) and matrix.iloc[0, 2] == pytest.approx(12.3)
    assert pixel.decode().splitlines()[1] == "0,0.0,,12.3"
    xyz = xrd.table_csv(tables["pixel_xyz"]).decode().splitlines()
    assert xyz[:2] == ["x_pixel,y_pixel,intensity", "0,0,0.0"] and len(xyz) == 6

    q = xrd.table_csv(xrd.q_tables(GRID, ["matrix"])["q"]).decode().splitlines()
    assert q == [",0.37,0.374", "0.922,1.0,", "0.918,2.0,3.0"]


def test_xlsx_layout(tmp_path):
    sheets = [
        ("第1張_像素", xrd.pixel_tables(VALUES, ["matrix"])["pixel"]),
        ("第1張_q", xrd.q_tables(GRID, ["matrix"])["q"]),
        ("第1張_q_XYZ", xrd.q_tables(GRID, ["xyz"])["q_xyz"]),
    ]
    path = tmp_path / "out.xlsx"
    assert xrd.write_xlsx(path, sheets, ["說明", "line"]) == [s for s, _ in sheets]
    book = pd.read_excel(path, sheet_name=None, header=None)
    assert list(book) == ["第1張_像素", "第1張_q", "第1張_q_XYZ", "說明"]
    pixel = book["第1張_像素"]
    assert pixel.shape == (3, 4) and np.isnan(pixel.iloc[0, 0]) and pixel.iloc[0, 1:].tolist() == [0, 1, 2]
    assert pixel.iloc[1, 0] == 0 and np.isnan(pixel.iloc[1, 2]) and pixel.iloc[1, 3] == pytest.approx(12.3)
    q = book["第1張_q"]
    assert q.iloc[0, 1:].tolist() == [0.37, 0.374] and q.iloc[1:, 0].tolist() == [0.922, 0.918]
    assert book["第1張_q_XYZ"].values.tolist()[1:] == [[0.37, 0.922, 1.0], [0.37, 0.918, 2.0], [0.374, 0.918, 3.0]]


def test_xlsx_splits_long_xyz(tmp_path, monkeypatch):
    monkeypatch.setattr(xrd, "XYZ_ROWS_PER_SHEET", 2)
    table = xrd.pixel_tables(VALUES, ["xyz"])["pixel_xyz"]  # 5 valid rows -> 3 sheets
    names = xrd.write_xlsx(tmp_path / "out.xlsx", [("a_XYZ", table)], [])
    assert names == ["a_XYZ_1", "a_XYZ_2", "a_XYZ_3"]


def test_downsample_max_keeps_spots():
    values = np.zeros((10, 10))
    values[7, 3] = 9.0
    small, factor = xrd.downsample_max(values, target=5)
    assert factor == 2 and small.shape == (5, 5) and small[3, 1] == 9.0
