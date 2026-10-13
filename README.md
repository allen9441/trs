# 2D XRD H5 → Excel / CSV

把 Eiger 2D XRD 的 H5 檔轉成亮點圖表格，格式與 `50V_亮點圖_19_9-29_1-150_Origin.xlsx` 相同：
每個張數範圍輸出 **像素座標**（x pixel × y pixel × 強度）與 **q 座標**（Q(x) × Q(y) × 強度）兩種表格。

- 後端：FastAPI（`backend/`）— 讀 H5、計算亮點 z 值、換算 q、輸出 xlsx / CSV
- 前端：Streamlit（`frontend/app.py`）— 上傳檔案、設定、預覽、下載

## 使用方式

**資料集** = 一個 master 檔（`*_master.h5`）+ 它引用的資料檔（`*_data_00000x.h5`，檔名不可改）+ 選用的 pyFAI `.poni`。
上傳後會存在 `data/<資料集名稱>/`，之後直接選用，不用重傳：

```
data/
└── 25A/
    ├── 25A_master.h5
    └── 25A_data_000001.h5
```

1. 在「1. 選擇資料」：
   - **已儲存的資料集**分頁：選資料集 → 載入（也可刪除）。
   - **上傳新資料集**分頁：選 master + data（+ .poni）→ 輸入名稱（預設為 master 檔名去掉 `_master`）→ 上傳並儲存。同名時要勾選「覆蓋」。
   - 也可以自己把整個資料夾放進 `data/`，重新整理頁面就會出現。
2. 輸入張數範圍，例如 `19; 9-29; 1-150`（從 1 開始）。
3. 選擇輸出：
   - **Excel**（預設）：所有表格在同一個活頁簿，工作表 `第19張_像素`、`第19張_q`…，最後是 `說明`。
   - **CSV**：每個表格一個檔，打包成 zip。
   - 表格形式：矩陣（第一列 x、第一欄 y，可直接 Convert to Matrix 給 Origin），或三欄 XYZ。
     像素 XYZ 約 106 萬列，超過 Excel 單一工作表上限，會分成 `_XYZ_1`、`_XYZ_2`。

資料夾位置可用環境變數 `XRD_DATA` 改（預設為專案內的 `data/`）。
同一個資料集第二次開啟時會重用已算好的平均影像，轉換比第一次快。

## 計算方式

- 亮點強度 z = (該張計數 − 期望值) / √期望值；期望值 = 全部張平均影像 × 該張整體強度比例。負值設為 0，範圍內每個像素取最大值。
  另可選「原始計數（範圍加總）」。
- 空白格 = 無資料：偵測器遮罩／模組間隙、溢位（≥ count cutoff）、恆亮像素（平均 > 5×5 區域中位數 10 倍）。
- q：qx = |q|cosχ、qy = |q|sinχ，網格 0.004 Å⁻¹（可調），每格取最大 z；+qx = 影像右方，+qy = 影像上方。
- 幾何：預設用 H5 標頭（距離、光束中心、波長，**無傾斜、未精修**），可在頁面上手動修正；上傳 `.poni` 則用 pyFAI 校正（含傾斜），q 值較準確。

## 開發

```bash
uv sync
./run.sh                       # 後端 :8000 + 前端 :8501
uv run pytest                  # 測試（合成資料）
```

## 部署（pm2 + nginx）

`/home/user/builds/ecosystem.config.js` 中的兩個 app：

| pm2 名稱 | 內容 | 位址 |
|---|---|---|
| `trs-api` | FastAPI 後端 | 127.0.0.1:8508（僅本機） |
| `trs` | Streamlit 前端 | 0.0.0.0:8507，base path `/trs` |

nginx（`/home/user/builds/proxy/nginx.conf`）的 `location /trs/` 反代理到 8507，
並把上傳上限放寬到 4096M（資料檔數百 MB；Streamlit 端上限在 `.streamlit/config.toml`）。
`data/` 要讓執行 pm2 的使用者可寫入（目前為 `drwxrwsr-x`，群組 `user`）。

```bash
cd /home/user/builds
pm2 start ecosystem.config.js --only trs-api,trs && pm2 save
docker exec lab-nginx nginx -s reload
```
