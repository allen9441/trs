"""Streamlit frontend: upload 2D XRD H5 files, choose frame ranges, download Excel / CSV tables.

Run:  uv run streamlit run frontend/app.py   (backend URL via XRD_BACKEND, default http://localhost:8000)
"""

import os
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND = os.environ.get("XRD_BACKEND", "http://localhost:8000").rstrip("/")
TIMEOUT = 3600

# Sequential blue ramp (light -> dark); values below the lower bound render white.
SEQUENTIAL = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NO_DATA = "#d6d5d0"

st.set_page_config(page_title="2D XRD H5 → Excel / CSV", layout="wide")


def api(method: str, path: str, **kwargs) -> requests.Response:
    try:
        resp = requests.request(method, f"{BACKEND}{path}", timeout=TIMEOUT, **kwargs)
    except requests.ConnectionError:
        st.error(f"無法連線到後端 {BACKEND}，請先啟動：`uv run uvicorn backend.api:app --port 8000`")
        st.stop()
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        st.error(f"後端錯誤（{resp.status_code}）：{detail}")
        st.stop()
    return resp


def heatmap(preview: dict, title: str, x_title: str, y_title: str, reverse_y: bool, lo: float, hi: float | None):
    z = np.array([[np.nan if v is None else v for v in row] for row in preview["z"]], dtype=float)
    finite = z[np.isfinite(z)]
    hi = hi or (float(finite.max()) if finite.size else lo * 10)
    hi = max(hi, lo * 1.01)
    color = np.log10(np.clip(z, lo, hi))
    ticks = [v for v in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 1e4, 1e5, 1e6] if lo <= v <= hi]
    fig = go.Figure(go.Heatmap(
        x=preview["x"],
        y=preview["y"],
        z=color,
        customdata=z,
        zmin=np.log10(lo),
        zmax=np.log10(hi),
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
        colorbar=dict(title="強度", tickvals=np.log10(ticks), ticktext=[f"{t:g}" for t in ticks]),
        hovertemplate=f"{x_title}: %{{x}}<br>{y_title}: %{{y}}<br>強度: %{{customdata:.1f}}<extra></extra>",
        hoverongaps=False,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=NO_DATA,
    )
    fig.update_xaxes(title=x_title, showgrid=False, zeroline=False)
    fig.update_yaxes(
        title=y_title, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1,
        autorange="reversed" if reverse_y else True,
    )
    return fig


st.title("2D XRD H5 → Excel / CSV 轉換")
st.caption(
    "資料集 = 一個 Eiger master 檔（*_master.h5）加上它引用的資料檔（*_data_00000x.h5，檔名不可更改），"
    "可另含 pyFAI 校正檔 .poni 以取得精確的 q 座標（含偵測器傾斜）。上傳後會存在伺服器的 data/ 資料夾，之後直接選用、不必重傳。"
)


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def close_session() -> None:
    old = st.session_state.pop("session", None)
    if old:
        api("DELETE", f"/sessions/{old['session_id']}")
    st.session_state.pop("result", None)


def open_dataset(name: str) -> None:
    close_session()
    with st.spinner(f"讀取 {name}…"):
        st.session_state.session = api("POST", f"/datasets/{quote(name, safe='')}/session").json()


# ---------------------------------------------------------------- 1. data

st.header("1. 選擇資料")
datasets = api("GET", "/datasets").json()
current = (st.session_state.get("session") or {}).get("dataset")
tab_saved, tab_new = st.tabs([f"已儲存的資料集（{len(datasets)}）", "上傳新資料集"])

with tab_saved:
    if not datasets:
        st.info("還沒有儲存的資料集，請到「上傳新資料集」分頁上傳。")
    else:
        names = [d["name"] for d in datasets]
        pick = st.selectbox("資料集", names, index=names.index(current) if current in names else 0)
        d = next(x for x in datasets if x["name"] == pick)
        usable = not d.get("error") and not d.get("missing_files")
        if d.get("error"):
            st.error(d["error"])
        elif d["missing_files"]:
            st.warning("缺少 master 檔引用的資料檔：" + "、".join(d["missing_files"]))
        else:
            st.caption(
                f"{d['n_frames']} 張 · {d['shape'][0]} × {d['shape'][1]} pixel"
                + (f" · 校正檔 {d['poni']}" if d["poni"] else " · 無 .poni")
            )
        st.caption("檔案：" + "、".join(f"{f['name']}（{fmt_size(f['size'])}）" for f in d["files"]))
        load, delete, _ = st.columns([1, 1, 4])
        if load.button("載入", type="primary", disabled=not usable):
            open_dataset(pick)
        with delete.popover("刪除"):
            st.warning(f"永久刪除資料集 {pick}（含其中所有檔案）？")
            if st.button("確認刪除", key=f"del_{pick}"):
                api("DELETE", f"/datasets/{quote(pick, safe='')}")
                if current == pick:
                    st.session_state.pop("session", None)
                    st.session_state.pop("result", None)
                st.rerun()

with tab_new:
    uploads = st.file_uploader(
        "H5 檔案（master + data）與選用的 .poni",
        type=["h5", "hdf5", "nxs", "poni"],
        accept_multiple_files=True,
    )
    master_names = [f.name for f in uploads or [] if f.name.endswith("_master.h5")]
    default_name = master_names[0].removesuffix("_master.h5") if len(master_names) == 1 else ""
    name = st.text_input(
        "資料集名稱",
        value=default_name,
        help="檔案會存到 data/<名稱>/。留空則用 master 檔名去掉 _master。",
    )
    overwrite = st.checkbox("同名資料集已存在時覆蓋")
    if st.button("上傳並儲存", type="primary", disabled=not uploads):
        with st.spinner("上傳中…（大型資料檔需要一些時間）"):
            files = [("files", (f.name, f, "application/octet-stream")) for f in uploads]
            saved = api("POST", "/datasets", files=files, data={"name": name, "overwrite": str(overwrite).lower()}).json()
        open_dataset(saved["name"])
        st.success(f"已儲存為 data/{saved['name']}/，下次可直接從「已儲存的資料集」選用。")

session = st.session_state.get("session")
if not session:
    st.stop()
st.caption(f"目前使用：資料集 {session['dataset']}" if session.get("dataset") else "目前使用：單次上傳（未儲存）")

geo = session["header_geometry"]
cols = st.columns(4)
cols[0].metric("Master 檔", session["master"])
cols[1].metric("影像張數", session["n_frames"])
cols[2].metric("影像尺寸 (y × x)", f"{session['shape'][0]} × {session['shape'][1]}")
cols[3].metric("偵測器", session["detector"] or "—")
if session["missing_files"]:
    st.error("缺少 master 檔引用的資料檔，請補上後重新上傳（檔名需完全相同）：" + "、".join(session["missing_files"]))
    st.stop()

# ---------------------------------------------------------------- 2. settings

st.header("2. 轉換設定")
left, right = st.columns(2)
with left:
    n = session["n_frames"]
    frames = st.text_input(
        "張數範圍（從 1 開始；多組以分號分隔）",
        value=f"1-{n}",
        help="例：19; 9-29; 1-150。單張輸出該張；範圍內每個像素取最大值（亮點會累積成環）。",
    )
    split_label = st.radio(
        "輸出方式",
        ["疊加（每組範圍合成一張表）", "逐張（範圍內每一張各自一張表）", "兩者都要（疊加 + 逐張）"],
        help="疊加：範圍內每個像素取最大值（z）或加總（計數）。逐張：把範圍展開，每張影像單獨輸出表格。"
        "例：1-150 逐張 = 150 組表格。張數多時建議用 CSV，Excel 會非常大。",
    )
    split = ["range", "each", "both"][["疊加", "逐張", "兩者"].index(split_label[:2])]
    mode_label = st.radio(
        "強度",
        ["亮點強度 z（範圍取最大）", "原始計數（範圍加總）"],
        help="z = (計數 − 期望值) / √期望值；期望值 = 全部張平均影像 × 該張整體強度比例。負值設為 0，z ≥ 8 幾乎不可能是雜訊。",
    )
    mode = "z" if mode_label.startswith("亮點") else "counts"
    output = st.radio(
        "輸出檔案",
        ["xlsx", "csv"],
        format_func={"xlsx": "Excel 活頁簿（所有表格合併成一個檔，每個表格一個工作表）", "csv": "CSV（每個表格一個檔，打包成 zip）"}.get,
    )
    formats = st.multiselect(
        "表格形式",
        options=["matrix", "xyz"],
        default=["matrix"],
        format_func={"matrix": "矩陣（第一列 x、第一欄 y，可直接給 Origin）", "xyz": "三欄 XYZ（x, y, 強度）"}.get,
        help="每組（疊加範圍或單張）輸出像素座標與 q 座標兩種表格。像素 XYZ 約 106 萬列，超過 Excel 單一工作表上限，會分成多個工作表。",
    )
    q_bin = st.number_input("q 網格間距 (Å⁻¹)", min_value=0.0005, max_value=0.1, value=0.004, step=0.001, format="%.4f")

with right:
    sources = ["poni", "header"] if session["poni"] else ["header"]
    source = st.radio(
        "q 換算幾何",
        sources,
        format_func={"poni": f"校正檔 {session['poni']}（建議）", "header": "H5 標頭（無傾斜，可手動修正）"}.get,
    )
    geometry = {"source": source}
    if source == "header":
        if not session["poni"]:
            st.info("H5 標頭的幾何未經精修、不含偵測器傾斜；需要精確 q 值時請一起上傳 .poni 校正檔。")
        c1, c2 = st.columns(2)
        geometry["distance_mm"] = c1.number_input("樣品–偵測器距離 (mm)", value=float(geo["distance_mm"] or 0), format="%.3f")
        geometry["wavelength_A"] = c2.number_input("波長 (Å)", value=float(geo["wavelength_A"] or 0), format="%.6f")
        geometry["center_x"] = c1.number_input("光束中心 x (pixel)", value=float(geo["center_x"] or 0), format="%.2f")
        geometry["center_y"] = c2.number_input("光束中心 y (pixel)", value=float(geo["center_y"] or 0), format="%.2f")
        st.caption(f"像素尺寸 {geo['pixel_um']} µm。+qx = 影像右方，+qy = 影像上方。")

if st.button("開始轉換", type="primary", disabled=not formats):
    body = {"frames": frames, "mode": mode, "formats": formats, "output": output, "split": split, "q_bin": q_bin, "geometry": geometry}
    with st.spinner("計算中…（第一次轉換需先統計全部張的平均影像；Excel 輸出需要較久）"):
        result = api("POST", f"/sessions/{session['session_id']}/convert", json=body).json()
        data = api("GET", f"/sessions/{session['session_id']}/results/{result['result_id']}").content
    st.session_state.result = {**result, "mode": mode, "data": data}

# ---------------------------------------------------------------- 3. results

result = st.session_state.get("result")
if not result:
    st.stop()

st.header("3. 結果")
is_xlsx = result["file_name"].endswith(".xlsx")
st.download_button(
    f"下載 {result['file_name']}",
    data=result["data"],
    file_name=result["file_name"],
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if is_xlsx else "application/zip",
    type="primary",
)
st.caption(("工作表：" if is_xlsx else "檔案：") + "、".join(result["contents"]) + (" 、說明" if is_xlsx else "、說明.txt"))

is_z = result["mode"] == "z"
summary = pd.DataFrame([
    {
        "張數": r["label"],
        "最大強度": r["max"],
        **({f"z ≥ {result['spot_threshold']:g} 像素數": r["n_bright"]} if is_z else {}),
    }
    for r in result["ranges"]
])
st.dataframe(summary, hide_index=True)
ex = result["excluded_pixels"]
qr = result["q_range"]
st.caption(
    f"空白格 = 無資料：遮罩／模組間隙 {ex['masked']}、溢位 {ex['overflow']}、恆亮 {ex['hot']} 像素。"
    f"q 範圍：qx {qr['qx'][0]:.3f} ~ {qr['qx'][1]:.3f}，qy {qr['qy'][0]:.3f} ~ {qr['qy'][1]:.3f} Å⁻¹。"
)

st.subheader("預覽")
c1, c2 = st.columns(2)
default_lo = result["spot_threshold"] if is_z else 1.0
lo = c1.number_input("色階下限（低於此值為白色）", min_value=0.01, value=float(default_lo))
hi = c2.number_input("色階上限（0 = 自動）", min_value=0.0, value=0.0)
st.caption("對數色階；灰色 = 無資料。預覽為降採樣（每格取區塊最大值），CSV 為完整解析度。")

shown = result["ranges"]
if len(shown) > 12:
    labels = [r["label"] for r in shown]
    picked = st.multiselect("預覽哪幾組（下載檔含全部）", labels, default=labels[:6])
    shown = [r for r in shown if r["label"] in picked]
for r in shown:
    st.markdown(f"**第 {r['label']} 張**")
    p1, p2 = st.columns(2)
    pp, pq = r["preview_pixel"], r["preview_q"]
    p1.plotly_chart(
        heatmap(pp, f"像素座標（每 {pp['factor']}×{pp['factor']} 取最大）", "x (pixel)", "y (pixel)", True, lo, hi or None),
        width="stretch",
    )
    p2.plotly_chart(
        heatmap(pq, f"q 座標（每 {pq['factor']}×{pq['factor']} 取最大）", "qx (Å⁻¹)", "qy (Å⁻¹)", False, lo, hi or None),
        width="stretch",
    )
