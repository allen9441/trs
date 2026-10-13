"""Core processing: Eiger/NeXus H5 frames -> bright-spot map -> pixel / q tables.

Bright-spot intensity (z) follows the method of the reference workbook:
    expected_k = mean image over all frames x (total of frame k / total of mean image)
    z = (counts_k - expected_k) / sqrt(expected_k), negatives set to 0
Over a frame range each pixel keeps its maximum z. Invalid pixels (detector mask,
module gaps, overflow, hot pixels) are NaN and become empty cells in the CSV.

q mapping: qx = |q| cos(chi), qy = |q| sin(chi); +qx = image right, +qy = image top.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import hdf5plugin  # noqa: F401  registers the bitshuffle/LZ4 filters used by Eiger data files
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

DETECTOR = "entry/instrument/detector"
SPECIFIC = f"{DETECTOR}/detectorSpecific"

HOT_PIXEL_RATIO = 10.0  # mean image > 10x its 5x5 local median -> hot pixel
INTENSITY_DECIMALS = 1


# ---------------------------------------------------------------- header / frames


@dataclass
class Header:
    n_frames: int
    height: int
    width: int
    dtype: str
    wavelength_A: float | None
    distance_m: float | None
    beam_center_x: float | None
    beam_center_y: float | None
    pixel_x_m: float
    pixel_y_m: float
    count_cutoff: float | None
    description: str


def _read(f: h5py.File, path: str, default=None):
    if path not in f:
        return default
    value = f[path][()]
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


class FrameSource:
    """Frames of a master file, possibly spread over external data files (1-based numbering)."""

    def __init__(self, master: str | Path):
        self.path = Path(master)
        self.blocks: list[tuple[str, int]] = []
        self.missing_files: list[str] = []
        with h5py.File(self.path, "r") as f:
            if "entry/data" not in f:
                raise ValueError(f"{self.path.name}: 找不到 entry/data，不是 NeXus/Eiger master 檔")
            group = f["entry/data"]
            for name in sorted(group.keys()):
                link = group.get(name, getlink=True)
                if isinstance(link, h5py.ExternalLink) and not (self.path.parent / link.filename).exists():
                    self.missing_files.append(link.filename)
                    continue
                ds = group[name]
                if isinstance(ds, h5py.Dataset) and ds.ndim in (2, 3):
                    self.blocks.append((name, 1 if ds.ndim == 2 else ds.shape[0]))
                    self.frame_shape = tuple(ds.shape[-2:])
                    self.dtype = ds.dtype
        if not self.blocks and not self.missing_files:
            raise ValueError(f"{self.path.name}: entry/data 內沒有影像資料")

    @property
    def n_frames(self) -> int:
        return sum(n for _, n in self.blocks)

    def iter_frames(self, numbers=None):
        """Yield (frame_number, 2D array) for the requested 1-based frame numbers (all if None)."""
        if self.missing_files:
            raise FileNotFoundError("缺少資料檔：" + ", ".join(self.missing_files))
        wanted = None if numbers is None else set(numbers)
        with h5py.File(self.path, "r") as f:
            first = 1
            for name, n in self.blocks:
                ds = f["entry/data"][name]
                for i in range(n):
                    number = first + i
                    if wanted is None or number in wanted:
                        yield number, ds[()] if ds.ndim == 2 else ds[i]
                first += n


def read_header(master: str | Path, source: FrameSource) -> Header:
    with h5py.File(master, "r") as f:
        height, width = source.frame_shape if source.blocks else (
            _read(f, f"{SPECIFIC}/y_pixels_in_detector"),
            _read(f, f"{SPECIFIC}/x_pixels_in_detector"),
        )
        return Header(
            n_frames=source.n_frames,
            height=int(height or 0),  # unknown until the data files are present
            width=int(width or 0),
            dtype=str(source.dtype) if source.blocks else "",
            wavelength_A=_read(f, "entry/instrument/beam/incident_wavelength"),
            distance_m=_read(f, f"{DETECTOR}/detector_distance"),
            beam_center_x=_read(f, f"{DETECTOR}/beam_center_x"),
            beam_center_y=_read(f, f"{DETECTOR}/beam_center_y"),
            pixel_x_m=_read(f, f"{DETECTOR}/x_pixel_size", 75e-6),
            pixel_y_m=_read(f, f"{DETECTOR}/y_pixel_size", 75e-6),
            count_cutoff=_read(f, f"{SPECIFIC}/countrate_correction_count_cutoff"),
            description=_read(f, f"{DETECTOR}/description", ""),
        )


def read_pixel_mask(master: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """True where the detector flags the pixel (gaps, dead, hot...)."""
    with h5py.File(master, "r") as f:
        mask = _read(f, f"{SPECIFIC}/pixel_mask")
    if mask is None or mask.shape != shape:
        return np.zeros(shape, dtype=bool)
    return mask != 0


# ---------------------------------------------------------------- statistics / intensity


@dataclass
class Stats:
    mean: np.ndarray  # mean image over all frames
    valid: np.ndarray  # usable pixels
    scale: np.ndarray  # per-frame total / mean total (index = frame number - 1)
    n_masked: int
    n_overflow: int
    n_hot: int


def compute_stats(source: FrameSource, header: Header, mask: np.ndarray) -> Stats:
    cutoff = header.count_cutoff
    if cutoff is None:
        cutoff = np.iinfo(source.dtype).max // 2 if np.issubdtype(source.dtype, np.integer) else np.inf

    total = np.zeros(mask.shape)
    overflow = np.zeros(mask.shape, dtype=bool)
    for _, frame in source.iter_frames():
        overflow |= frame >= cutoff
        total += frame
    mean = total / source.n_frames

    valid = ~mask & ~overflow
    local = median_filter(np.where(valid, mean, 0.0), size=5)
    hot = valid & (mean > HOT_PIXEL_RATIO * np.maximum(local, 1.0))
    valid &= ~hot

    mean_total = mean[valid].sum()
    scale = np.empty(source.n_frames)
    for number, frame in source.iter_frames():
        scale[number - 1] = frame[valid].sum(dtype=np.float64) / mean_total if mean_total > 0 else 1.0

    return Stats(
        mean=mean,
        valid=valid,
        scale=scale,
        n_masked=int(mask.sum()),
        n_overflow=int((overflow & ~mask).sum()),
        n_hot=int(hot.sum()),
    )


def intensity_map(source: FrameSource, stats: Stats, frames: list[int], mode: str = "z") -> np.ndarray:
    """mode "z": max bright-spot z over frames; mode "counts": summed raw counts. NaN = invalid."""
    out = np.zeros(stats.mean.shape)
    if mode == "z":
        for number, frame in source.iter_frames(frames):
            expected = stats.mean * stats.scale[number - 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(expected > 0, (frame - expected) / np.sqrt(expected), 0.0)
            np.maximum(out, z, out=out)
    elif mode == "counts":
        for _, frame in source.iter_frames(frames):
            out += frame
    else:
        raise ValueError(f"unknown mode: {mode}")
    out[~stats.valid] = np.nan
    return out


# ---------------------------------------------------------------- geometry / q


def header_integrator(
    header: Header,
    distance_m: float | None = None,
    center_x: float | None = None,
    center_y: float | None = None,
    wavelength_A: float | None = None,
):
    """pyFAI integrator from the H5 header (no tilt); any argument overrides the header value."""
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    distance_m = distance_m if distance_m is not None else header.distance_m
    center_x = center_x if center_x is not None else header.beam_center_x
    center_y = center_y if center_y is not None else header.beam_center_y
    wavelength_A = wavelength_A if wavelength_A is not None else header.wavelength_A
    if None in (distance_m, center_x, center_y, wavelength_A):
        raise ValueError("H5 標頭缺少幾何資訊（距離／光束中心／波長），請手動輸入或上傳 .poni")

    detector = Detector(pixel1=header.pixel_y_m, pixel2=header.pixel_x_m, max_shape=(header.height, header.width))
    ai = AzimuthalIntegrator(detector=detector)
    ai.setFit2D(distance_m * 1e3, center_x, center_y)
    ai.wavelength = wavelength_A * 1e-10
    return ai


def poni_integrator(path: str | Path, header: Header):
    """pyFAI integrator from a .poni calibration file (includes detector tilt)."""
    import pyFAI

    ai = pyFAI.load(str(path))
    shape = getattr(ai.detector, "max_shape", None)
    if shape is not None and tuple(shape) != (header.height, header.width):
        raise ValueError(f".poni 偵測器尺寸 {tuple(shape)} 與影像 {(header.height, header.width)} 不符")
    if not ai.wavelength:
        if header.wavelength_A is None:
            raise ValueError(".poni 與 H5 都沒有波長資訊")
        ai.wavelength = header.wavelength_A * 1e-10
    return ai


def q_components(ai, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel (qx, qy) in 1/Å; +qx = image right (increasing column), +qy = image top."""
    d1, d2 = np.indices(shape, dtype=np.float64)
    z, y, x = ai.calc_pos_zyx(d1=d1, d2=d2, corners=False)
    r = np.hypot(x, y)
    q = 4 * np.pi * np.sin(np.arctan2(r, z) / 2) / (ai.wavelength * 1e10)
    with np.errstate(divide="ignore", invalid="ignore"):
        qx = np.where(r > 0, q * x / r, 0.0)
        qy = np.where(r > 0, -q * y / r, 0.0)
    return qx, qy


@dataclass
class QGrid:
    qx: np.ndarray  # bin centers, ascending (columns)
    qy: np.ndarray  # bin centers, descending (rows, top = largest qy)
    values: np.ndarray  # (len(qy), len(qx)), NaN = empty bin


def bin_to_q(values: np.ndarray, qx: np.ndarray, qy: np.ndarray, step: float, agg: str = "max") -> QGrid:
    ok = np.isfinite(values)
    v, ix, iy = values[ok], np.floor(qx[ok] / step).astype(np.int64), np.floor(qy[ok] / step).astype(np.int64)
    if v.size == 0:
        return QGrid(np.array([]), np.array([]), np.empty((0, 0)))
    ix0, iy0 = ix.min(), iy.min()
    nx, ny = ix.max() - ix0 + 1, iy.max() - iy0 + 1
    flat = (iy - iy0) * nx + (ix - ix0)
    if agg == "max":
        grid = np.full(nx * ny, -np.inf)
        np.maximum.at(grid, flat, v)
        grid[np.isneginf(grid)] = np.nan
    else:
        counts = np.bincount(flat, minlength=nx * ny)
        with np.errstate(invalid="ignore"):
            grid = np.bincount(flat, weights=v, minlength=nx * ny) / counts
    grid = grid.reshape(ny, nx)[::-1]
    return QGrid(
        qx=(ix0 + np.arange(nx) + 0.5) * step,
        qy=((iy0 + np.arange(ny) + 0.5) * step)[::-1],
        values=grid,
    )


# ---------------------------------------------------------------- frame ranges


@dataclass(frozen=True)
class FrameRange:
    start: int
    end: int

    @property
    def label(self) -> str:
        return str(self.start) if self.start == self.end else f"{self.start}-{self.end}"

    @property
    def frames(self) -> list[int]:
        return list(range(self.start, self.end + 1))


def parse_ranges(text: str, n_frames: int) -> list[FrameRange]:
    """Parse e.g. "19; 9-29, 1-150" (1-based, inclusive)."""
    ranges = []
    for part in re.split(r"[;,，；\s]+", text.strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:\s*[-~～]\s*(\d+))?", part)
        if not m:
            raise ValueError(f"無法解讀張數範圍：{part!r}（格式例：19 或 9-29）")
        start, end = int(m[1]), int(m[2] or m[1])
        if start > end:
            start, end = end, start
        if start < 1 or end > n_frames:
            raise ValueError(f"範圍 {part} 超出 1-{n_frames}")
        ranges.append(FrameRange(start, end))
    if not ranges:
        raise ValueError("請輸入至少一個張數範圍")
    return list(dict.fromkeys(ranges))


# ---------------------------------------------------------------- tables / output

EXCEL_MAX_ROWS = 1_048_576
XYZ_ROWS_PER_SHEET = 1_000_000  # pixel XYZ (~1.06 M rows) exceeds one Excel sheet -> split


@dataclass
class Table:
    """kind "matrix": values (ny, nx), first row = x labels, first column = y labels (Origin-ready).
    kind "xyz": values (n, 3) with column names; NaN cells/rows (no data) are left out."""

    kind: str
    values: np.ndarray
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    columns: tuple[str, str, str] | None = None


def _xyz(x: np.ndarray, y: np.ndarray, values: np.ndarray, names: tuple[str, str, str]) -> Table:
    ok = np.isfinite(values)
    return Table("xyz", np.column_stack([x[ok], y[ok], values[ok]]), columns=names)


def pixel_tables(values: np.ndarray, formats: list[str]) -> dict[str, Table]:
    """Keys "pixel" / "pixel_xyz"; x = column index, y = row index (image top = 0)."""
    h, w = values.shape
    out = {}
    if "matrix" in formats:
        out["pixel"] = Table("matrix", values, x=np.arange(w), y=np.arange(h))
    if "xyz" in formats:
        yy, xx = np.indices(values.shape)
        out["pixel_xyz"] = _xyz(xx, yy, values, ("x_pixel", "y_pixel", "intensity"))
    return out


def q_tables(grid: QGrid, formats: list[str]) -> dict[str, Table]:
    """Keys "q" / "q_xyz"; rows top-down by descending qy as in the reference workbook."""
    out = {}
    if "matrix" in formats:
        out["q"] = Table("matrix", grid.values, x=np.round(grid.qx, 6), y=np.round(grid.qy, 6))
    if "xyz" in formats:
        qx, qy = np.meshgrid(np.round(grid.qx, 6), np.round(grid.qy, 6))
        out["q_xyz"] = _xyz(qx, qy, grid.values, ("qx_1/A", "qy_1/A", "intensity"))
    return out


def _label(value: float) -> str:
    return format(round(float(value), 6), "g")


def table_csv(table: Table) -> bytes:
    fmt = f"%.{INTENSITY_DECIMALS}f"
    if table.kind == "matrix":
        df = pd.DataFrame(table.values, index=[_label(v) for v in table.y], columns=[_label(v) for v in table.x])
        return df.to_csv(float_format=fmt, na_rep="", index_label="").encode()
    v = table.values
    df = pd.DataFrame({
        table.columns[0]: [_label(a) for a in v[:, 0]],
        table.columns[1]: [_label(a) for a in v[:, 1]],
        table.columns[2]: v[:, 2],
    })
    return df.to_csv(index=False, float_format=fmt).encode()


def _cells(row: np.ndarray) -> list:
    # None -> no cell written (blank), matching "empty = no data" in the reference workbook.
    return [None if v != v else v for v in np.round(row.astype(np.float64), INTENSITY_DECIMALS).tolist()]


def write_xlsx(path: str | Path, sheets: list[tuple[str, Table]], notes: list[str]) -> list[str]:
    """Write every table to its own sheet (plus a 說明 sheet); returns the sheet names written."""
    import xlsxwriter

    names = []
    with xlsxwriter.Workbook(str(path), {"constant_memory": True}) as wb:
        for name, table in sheets:
            if table.kind == "matrix":
                if table.values.shape[0] + 1 > EXCEL_MAX_ROWS or table.values.shape[1] + 1 > 16_384:
                    raise ValueError(f"{name}: 表格 {table.values.shape} 超過 Excel 上限，請改用 CSV 或加大 q 網格")
                ws = wb.add_worksheet(name)
                ws.write_row(0, 1, table.x.tolist())
                for i, (label, row) in enumerate(zip(table.y.tolist(), table.values), start=1):
                    ws.write_number(i, 0, label)
                    ws.write_row(i, 1, _cells(row))
                names.append(name)
                continue
            parts = range(0, max(len(table.values), 1), XYZ_ROWS_PER_SHEET)
            for part, start in enumerate(parts, start=1):
                sheet = name if len(parts) == 1 else f"{name}_{part}"
                ws = wb.add_worksheet(sheet)
                ws.write_row(0, 0, list(table.columns))
                chunk = table.values[start : start + XYZ_ROWS_PER_SHEET]
                for i, (a, b, c) in enumerate(chunk.tolist(), start=1):
                    ws.write_number(i, 0, a)
                    ws.write_number(i, 1, b)
                    ws.write_number(i, 2, round(c, INTENSITY_DECIMALS))
                names.append(sheet)
        ws = wb.add_worksheet("說明")
        ws.set_column(0, 0, 120)
        for i, line in enumerate(notes):
            ws.write_string(i, 0, line)
    return names


# ---------------------------------------------------------------- preview


def downsample_max(values: np.ndarray, target: int = 300) -> tuple[np.ndarray, int]:
    """Block-max downsample so the longer side is <= target (keeps 1-2 px spots visible)."""
    factor = max(1, int(np.ceil(max(values.shape) / target)))
    if factor == 1:
        return values, 1
    h, w = values.shape
    padded = np.full((-(-h // factor) * factor, -(-w // factor) * factor), np.nan)
    padded[:h, :w] = values
    blocks = padded.reshape(padded.shape[0] // factor, factor, padded.shape[1] // factor, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(blocks, axis=(1, 3)), factor
