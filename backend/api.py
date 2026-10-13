"""FastAPI backend: saved datasets (data/<name>/) or one-off uploads -> convert frame ranges -> download xlsx / CSV zip.

Run:  uv run uvicorn backend.api:app --port 8000
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import xrd

STORE = Path(os.environ.get("XRD_STORE", Path(tempfile.gettempdir()) / "xrd_h5_csv"))  # per-session results, wiped on restart
DATA_DIR = Path(os.environ.get("XRD_DATA", Path(__file__).resolve().parent.parent / "data"))  # saved datasets, kept
SPOT_THRESHOLD = 8.0  # z >= 8 is practically never noise; reported as the bright-pixel count

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Per-session results only live for the lifetime of the server process.
    shutil.rmtree(STORE, ignore_errors=True)
    STORE.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(STORE, ignore_errors=True)


app = FastAPI(title="2D XRD H5 → Excel / CSV", lifespan=lifespan)


@dataclass
class Session:
    id: str
    dir: Path
    master: Path
    poni: Path | None
    source: xrd.FrameSource
    header: xrd.Header
    lock: threading.Lock = field(default_factory=threading.Lock)
    stats: xrd.Stats | None = None
    q_cache: dict = field(default_factory=dict)
    results: dict[str, tuple[Path, str]] = field(default_factory=dict)  # result id -> (path, download name)
    dataset: str | None = None  # name under data/ when opened from a saved dataset

    def get_stats(self) -> xrd.Stats:
        with self.lock:
            if self.stats is None:
                # Mean image / hot pixels depend only on the data, so reopening a dataset reuses them.
                key = (str(self.master), self.master.stat().st_mtime_ns)
                if key not in _STATS:
                    mask = xrd.read_pixel_mask(self.master, (self.header.height, self.header.width))
                    _STATS[key] = xrd.compute_stats(self.source, self.header, mask)
                    while len(_STATS) > STATS_CACHE_SIZE:
                        _STATS.pop(next(iter(_STATS)))
                self.stats = _STATS[key]
            return self.stats

    def get_q(self, geometry: GeometryIn) -> tuple[np.ndarray, np.ndarray]:
        key = geometry.model_dump_json()
        with self.lock:
            if key not in self.q_cache:
                if geometry.source == "poni":
                    if self.poni is None:
                        raise ValueError("沒有上傳 .poni 檔")
                    ai = xrd.poni_integrator(self.poni, self.header)
                else:
                    ai = xrd.header_integrator(
                        self.header,
                        distance_m=None if geometry.distance_mm is None else geometry.distance_mm / 1e3,
                        center_x=geometry.center_x,
                        center_y=geometry.center_y,
                        wavelength_A=geometry.wavelength_A,
                    )
                self.q_cache = {key: xrd.q_components(ai, (self.header.height, self.header.width))}
            return self.q_cache[key]


sessions: dict[str, Session] = {}
_STATS: dict[tuple[str, int], xrd.Stats] = {}
STATS_CACHE_SIZE = 4


class GeometryIn(BaseModel):
    source: Literal["header", "poni"] = "header"
    distance_mm: float | None = Field(None, gt=0)
    center_x: float | None = None
    center_y: float | None = None
    wavelength_A: float | None = Field(None, gt=0)


class ConvertIn(BaseModel):
    frames: str
    mode: Literal["z", "counts"] = "z"
    formats: list[Literal["matrix", "xyz"]] = Field(default_factory=lambda: ["matrix"], min_length=1)
    output: Literal["xlsx", "csv"] = "xlsx"  # xlsx: one workbook, one sheet per table; csv: zip
    q_bin: float = Field(0.004, gt=0, le=0.1)
    geometry: GeometryIn = Field(default_factory=GeometryIn)


def _session(session_id: str) -> Session:
    if session_id not in sessions:
        raise HTTPException(404, "找不到工作階段，請重新上傳檔案")
    return sessions[session_id]


def _safe_name(name: str | None) -> str:
    # Keep the original file name: the master file references its data files by name.
    name = Path(name or "").name
    if not name or name.startswith(".") or not re.fullmatch(r"[\w.\-一-鿿 ]+", name):
        raise HTTPException(400, f"不合法的檔名：{name!r}")
    return name


def _is_master(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as f:
            return "entry/data" in f and "entry/instrument" in f
    except OSError:
        return False


def _session_info(s: Session) -> dict:
    h = s.header
    return {
        "session_id": s.id,
        "dataset": s.dataset,
        "master": s.master.name,
        "poni": s.poni.name if s.poni else None,
        "missing_files": s.source.missing_files,
        "n_frames": h.n_frames,
        "shape": [h.height, h.width],
        "dtype": h.dtype,
        "detector": h.description,
        "header_geometry": {
            "distance_mm": None if h.distance_m is None else round(h.distance_m * 1e3, 4),
            "center_x": h.beam_center_x,
            "center_y": h.beam_center_y,
            "wavelength_A": h.wavelength_A,
            "pixel_um": round(h.pixel_x_m * 1e6, 3),
        },
    }


def _preview(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    small, factor = xrd.downsample_max(values)
    xs = x[::factor][: small.shape[1]]
    ys = y[::factor][: small.shape[0]]
    rounded = np.round(small, 1)
    return {
        "x": np.round(xs, 6).tolist(),
        "y": np.round(ys, 6).tolist(),
        "z": [[None if np.isnan(v) else float(v) for v in row] for row in rounded],
        "factor": factor,
    }


SHEET_KIND = {"pixel": "像素", "q": "q", "pixel_xyz": "像素_XYZ", "q_xyz": "q_XYZ"}


def _sheet_name(mode: str, label: str, kind: str) -> str:
    # Same naming as the reference workbook: 第19張_像素, 第9-29張_q, ...
    prefix = "" if mode == "z" else "原始計數_"
    return f"{prefix}第{label}張_{SHEET_KIND[kind]}"


def _notes(session: Session, req: ConvertIn, stats: xrd.Stats, items: list[dict]) -> list[str]:
    h, g = session.header, req.geometry
    if g.source == "poni":
        geometry = f"幾何：{session.poni.name}（pyFAI 校正檔，含偵測器傾斜）。"
    else:
        distance = g.distance_mm if g.distance_mm is not None else h.distance_m * 1e3
        cx = g.center_x if g.center_x is not None else h.beam_center_x
        cy = g.center_y if g.center_y is not None else h.beam_center_y
        wl = g.wavelength_A if g.wavelength_A is not None else h.wavelength_A
        geometry = (
            f"幾何：H5 標頭（無傾斜）距離 {distance:.3f} mm、光束中心 x={cx:.2f}, y={cy:.2f} pixel、"
            f"波長 {wl:.6f} Å、像素 {h.pixel_x_m * 1e6:.0f} µm。"
        )
    if req.mode == "z":
        intensity = (
            "強度 = 亮點 z 值：z = (該張計數 − 期望值) / √期望值；期望值 = 全部張平均影像 × 該張整體強度比例。"
            "非晶散射等穩定訊號被扣掉，只剩瞬間的 Bragg 亮點。負值設為 0；z ≥ 8 幾乎不可能是雜訊。"
            "範圍：每個像素取範圍內最大的 z。"
        )
        q_agg = "每格取格內最大的 z"
    else:
        intensity = "強度 = 原始計數，範圍內各張加總。"
        q_agg = "每格取格內平均"
    lines = [
        "說明",
        f"來源：{session.master.name}（共 {h.n_frames} 張，{h.height} × {h.width} pixel）。張數從 1 開始數。",
        geometry,
        intensity,
    ]
    for item in items:
        line = f"第{item['label']}張：最大強度 {item['max']}"
        if item["n_bright"] is not None:
            line += f"；z ≥ {SPOT_THRESHOLD:g} 的像素 {item['n_bright']} 個"
        lines.append(line + "。")
    lines += [
        f"空白格 = 沒有數據：遮罩／模組間隙 {stats.n_masked}、溢位 {stats.n_overflow}、恆亮 {stats.n_hot} 像素。",
        "_像素 工作表：第一列 = x (pixel)，第一欄 = y (pixel)，y = 0 為影像上方。",
        f"_q 工作表：qx = |q|cosχ、qy = |q|sinχ（Å⁻¹），網格 {req.q_bin:g} Å⁻¹，標示為格中心；{q_agg}。"
        "qx 正方向 = 影像右方，qy 正方向 = 影像上方。",
    ]
    if "xyz" in req.formats:
        lines.append(
            f"_XYZ 工作表：三欄 (x, y, 強度)，只列出有數據的格；超過 {xrd.XYZ_ROWS_PER_SHEET:,} 列時分成 _1、_2… 多個工作表。"
        )
    lines += [
        "",
        "【Origin 作法（矩陣）】",
        "1. Data → Import From File → Excel，選要畫的工作表；Header Lines 設為 0。",
        "2. 選取整張工作表 → Worksheet → Convert to Matrix → Direct；勾選 X Values in First Row、Y Values in First Column。",
        "3. 選取矩陣 → Plot → Contour → Heatmap（或 Image Plot）。",
        "4. 像素座標的圖：Y 軸要反轉（Axis → Scale → Reverse），影像上方才會在上面；q 座標的圖不用反轉。",
    ]
    return lines


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _save_uploads(files: list[UploadFile], folder: Path) -> list[Path]:
    saved = []
    for upload in files:
        target = folder / _safe_name(upload.filename)
        with target.open("wb") as out:
            shutil.copyfileobj(upload.file, out, 16 * 1024 * 1024)
        saved.append(target)
    return saved


def _classify(saved: list[Path]) -> tuple[Path, Path | None]:
    """Pick the master H5 and the optional .poni out of a set of files."""
    ponis = [p for p in saved if p.suffix.lower() == ".poni"]
    masters = [p for p in saved if p.suffix.lower() != ".poni" and _is_master(p)]
    if len(masters) != 1:
        found = ", ".join(p.name for p in masters) or "無"
        raise HTTPException(400, f"需要剛好一個 master H5 檔（含 entry/instrument），目前找到：{found}")
    if len(ponis) > 1:
        raise HTTPException(400, "最多上傳一個 .poni 檔")
    return masters[0], ponis[0] if ponis else None


def _open_session(master: Path, poni: Path | None, results_dir: Path, dataset: str | None = None) -> Session:
    """results_dir only holds this session's outputs; the H5 files may live elsewhere (data/)."""
    source = xrd.FrameSource(master)
    return Session(
        id=results_dir.name,
        dir=results_dir,
        master=master,
        poni=poni,
        source=source,
        header=xrd.read_header(master, source),
        dataset=dataset,
    )


def _register(session: Session) -> dict:
    sessions[session.id] = session
    return _session_info(session)


@app.post("/sessions")
def create_session(files: list[UploadFile] = File(...)) -> dict:
    """Upload files for a one-off session (nothing is kept in data/)."""
    folder = STORE / uuid.uuid4().hex
    folder.mkdir(parents=True)
    try:
        master, poni = _classify(_save_uploads(files, folder))
        session = _open_session(master, poni, folder)
    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    except (ValueError, OSError) as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc
    return _register(session)


# ---------------------------------------------------------------- saved datasets (data/<name>/)


def _dataset_dir(name: str) -> Path:
    if not re.fullmatch(r"[\w.\-一-鿿 ]+", name) or name.startswith("."):
        raise HTTPException(400, f"不合法的資料集名稱：{name!r}")
    return DATA_DIR / name


def _existing_dataset(name: str) -> Path:
    folder = _dataset_dir(name)
    if not folder.is_dir():
        raise HTTPException(404, f"找不到資料集：{name}")
    return folder


def _find_files(folder: Path) -> tuple[Path | None, Path | None]:
    files = sorted(p for p in folder.iterdir() if p.is_file())
    ponis = [p for p in files if p.suffix.lower() == ".poni"]
    masters = [p for p in files if p.suffix.lower() != ".poni" and _is_master(p)]
    return (masters[0] if masters else None), (ponis[0] if ponis else None)


def _dataset_info(folder: Path) -> dict:
    info = {
        "name": folder.name,
        "files": [{"name": p.name, "size": p.stat().st_size} for p in sorted(folder.iterdir()) if p.is_file()],
    }
    master, poni = _find_files(folder)
    if master is None:
        return {**info, "error": "資料夾內找不到 master H5 檔"}
    try:
        source = xrd.FrameSource(master)
        header = xrd.read_header(master, source)
    except (ValueError, OSError) as exc:
        return {**info, "error": str(exc)}
    return {
        **info,
        "master": master.name,
        "poni": poni.name if poni else None,
        "missing_files": source.missing_files,
        "n_frames": header.n_frames,
        "shape": [header.height, header.width],
    }


def _forget_dataset(folder: Path) -> None:
    """Drop sessions and cached statistics that point into a dataset about to change."""
    for sid in [s.id for s in sessions.values() if folder in s.master.parents]:
        shutil.rmtree(sessions.pop(sid).dir, ignore_errors=True)
    for key in [k for k in _STATS if Path(k[0]).parent == folder]:
        del _STATS[key]


@app.get("/datasets")
def list_datasets() -> list[dict]:
    if not DATA_DIR.is_dir():
        return []
    return [_dataset_info(p) for p in sorted(DATA_DIR.iterdir()) if p.is_dir() and not p.name.startswith(".")]


@app.post("/datasets")
def save_dataset(
    files: list[UploadFile] = File(...),
    name: str = Form(""),
    overwrite: bool = Form(False),
) -> dict:
    """Store uploaded files as data/<name>/ (default name: master file name without _master)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    staging = DATA_DIR / f".incoming-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        master, _ = _classify(_save_uploads(files, staging))
        name = name.strip() or master.stem.removesuffix("_master")
        target = _dataset_dir(name)
        problems = xrd.FrameSource(master).missing_files
        if problems:
            raise HTTPException(400, "缺少 master 檔引用的資料檔，請一起上傳：" + ", ".join(problems))
        if target.exists():
            if not overwrite:
                raise HTTPException(409, f"資料集 {name!r} 已存在；勾選「覆蓋」或換一個名稱")
            _forget_dataset(target)
            shutil.rmtree(target)
        staging.rename(target)
    except HTTPException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except (ValueError, OSError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc
    return _dataset_info(target)


@app.delete("/datasets/{name}")
def delete_dataset(name: str) -> dict:
    folder = _existing_dataset(name)
    _forget_dataset(folder)
    shutil.rmtree(folder)
    return {"deleted": name}


@app.post("/datasets/{name}/session")
def open_dataset(name: str) -> dict:
    """Open a saved dataset for conversion (no upload needed)."""
    folder = _existing_dataset(name)
    master, poni = _find_files(folder)
    if master is None:
        raise HTTPException(400, "資料夾內找不到 master H5 檔")
    results = STORE / uuid.uuid4().hex
    results.mkdir(parents=True)
    try:
        session = _open_session(master, poni, results, dataset=name)
    except (ValueError, OSError) as exc:
        shutil.rmtree(results, ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc
    return _register(session)


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    return _session_info(_session(session_id))


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    session = sessions.pop(session_id, None)
    if session:
        shutil.rmtree(session.dir, ignore_errors=True)
    return {"deleted": bool(session)}


@app.post("/sessions/{session_id}/convert")
def convert(session_id: str, req: ConvertIn) -> dict:
    session = _session(session_id)
    if session.source.missing_files:
        raise HTTPException(400, "缺少資料檔：" + ", ".join(session.source.missing_files))
    try:
        ranges = xrd.parse_ranges(req.frames, session.header.n_frames)
        qx, qy = session.get_q(req.geometry)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    stats = session.get_stats()
    stem = session.master.stem.removesuffix("_master")
    result_id = uuid.uuid4().hex[:12]
    items, sheets, csv_files = [], [], {}
    for r in ranges:
        values = xrd.intensity_map(session.source, stats, r.frames, req.mode)
        grid = xrd.bin_to_q(values, qx, qy, req.q_bin, agg="max" if req.mode == "z" else "mean")
        tables = {**xrd.pixel_tables(values, req.formats), **xrd.q_tables(grid, req.formats)}
        for kind in [k for k in SHEET_KIND if k in tables]:
            table = tables[kind]
            sheets.append((_sheet_name(req.mode, r.label, kind), table))
            if req.output == "csv":
                csv_files[f"{stem}_{req.mode}_frames{r.label}_{kind}.csv"] = xrd.table_csv(table)
        h, w = values.shape
        items.append({
            "label": r.label,
            "frames": [r.start, r.end],
            "max": None if np.all(np.isnan(values)) else round(float(np.nanmax(values)), 1),
            "n_bright": int((values >= SPOT_THRESHOLD).sum()) if req.mode == "z" else None,
            "preview_pixel": _preview(values, np.arange(w, dtype=float), np.arange(h, dtype=float)),
            "preview_q": _preview(grid.values, grid.qx, grid.qy),
        })

    notes = _notes(session, req, stats, items)
    if req.output == "xlsx":
        file_name = f"{stem}_{'亮點圖' if req.mode == 'z' else '原始計數'}_{'_'.join(r.label for r in ranges)}.xlsx"
        path = session.dir / f"result_{result_id}.xlsx"
        try:
            contents = xrd.write_xlsx(path, sheets, notes)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    else:
        file_name = f"{stem}_{req.mode}_csv.zip"
        path = session.dir / f"result_{result_id}.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in csv_files.items():
                zf.writestr(name, data)
            zf.writestr("說明.txt", "\n".join(notes).encode())
        contents = list(csv_files)
    session.results[result_id] = (path, file_name)

    return {
        "result_id": result_id,
        "file_name": file_name,
        "contents": contents,
        "spot_threshold": SPOT_THRESHOLD,
        "excluded_pixels": {"masked": stats.n_masked, "overflow": stats.n_overflow, "hot": stats.n_hot},
        "q_range": {"qx": [float(qx.min()), float(qx.max())], "qy": [float(qy.min()), float(qy.max())]},
        "ranges": items,
    }


@app.get("/sessions/{session_id}/results/{result_id}")
def download(session_id: str, result_id: str) -> FileResponse:
    session = _session(session_id)
    if result_id not in session.results:
        raise HTTPException(404, "找不到結果檔")
    path, file_name = session.results[result_id]
    media = "application/zip" if path.suffix == ".zip" else (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    return FileResponse(path, media_type=media, filename=file_name)
