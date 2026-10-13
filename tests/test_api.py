import io
import zipfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.test_xrd import H, N, W, make_dataset


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.api.STORE", tmp_path / "store")
    from backend.api import app

    with TestClient(app) as c:
        yield c


def upload(client, folder, names):
    files = [("files", (n, (folder / n).read_bytes())) for n in names]
    return client.post("/sessions", files=files)


def test_upload_convert_download(client, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_dataset(src)
    resp = upload(client, src, ["t_master.h5", "t_data_000001.h5"])
    assert resp.status_code == 200, resp.text
    info = resp.json()
    assert info["n_frames"] == N and info["missing_files"] == []

    body = {"frames": "3; 1-6", "formats": ["matrix", "xyz"]}
    resp = client.post(f"/sessions/{info['session_id']}/convert", json=body)
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert [r["label"] for r in result["ranges"]] == ["3", "1-6"]
    assert result["ranges"][0]["n_bright"] >= 1
    assert result["file_name"] == "t_亮點圖_3_1-6.xlsx"

    resp = client.get(f"/sessions/{info['session_id']}/results/{result['result_id']}")
    assert resp.status_code == 200
    book = pd.read_excel(io.BytesIO(resp.content), sheet_name=None, header=None)
    kinds = ["像素", "q", "像素_XYZ", "q_XYZ"]
    assert list(book) == [f"第{r}張_{k}" for r in ["3", "1-6"] for k in kinds] + ["說明"]
    assert book["第3張_像素"].shape == (H + 1, W + 1)

    body = {"frames": "2", "mode": "counts", "output": "csv"}
    result = client.post(f"/sessions/{info['session_id']}/convert", json=body).json()
    assert result["contents"] == ["t_counts_frames2_pixel.csv", "t_counts_frames2_q.csv"]
    resp = client.get(f"/sessions/{info['session_id']}/results/{result['result_id']}")
    names = set(zipfile.ZipFile(io.BytesIO(resp.content)).namelist())
    assert names == {"t_counts_frames2_pixel.csv", "t_counts_frames2_q.csv", "說明.txt"}


def test_missing_data_file_reported(client, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_dataset(src)
    info = upload(client, src, ["t_master.h5"]).json()
    assert info["missing_files"] == ["t_data_000001.h5"]
    resp = client.post(f"/sessions/{info['session_id']}/convert", json={"frames": "1"})
    assert resp.status_code == 400 and "t_data_000001.h5" in resp.json()["detail"]


def test_rejects_upload_without_master(client, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_dataset(src)
    assert upload(client, src, ["t_data_000001.h5"]).status_code == 400


def test_bad_frame_range(client, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_dataset(src)
    info = upload(client, src, ["t_master.h5", "t_data_000001.h5"]).json()
    resp = client.post(f"/sessions/{info['session_id']}/convert", json={"frames": "1-99"})
    assert resp.status_code == 400


@pytest.fixture
def dclient(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.api.STORE", tmp_path / "store")
    monkeypatch.setattr("backend.api.DATA_DIR", tmp_path / "data")
    from backend.api import app

    with TestClient(app) as c:
        yield c


def src_files(tmp_path, names=("t_master.h5", "t_data_000001.h5")):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    make_dataset(src)
    return [("files", (n, (src / n).read_bytes())) for n in names]


def test_dataset_save_list_open_convert(dclient, tmp_path):
    assert dclient.get("/datasets").json() == []
    saved = dclient.post("/datasets", files=src_files(tmp_path)).json()
    assert saved["name"] == "t" and saved["n_frames"] == N and saved["missing_files"] == []
    assert sorted(f["name"] for f in saved["files"]) == ["t_data_000001.h5", "t_master.h5"]
    assert (tmp_path / "data" / "t" / "t_master.h5").exists()
    assert not [p for p in (tmp_path / "data").iterdir() if p.name.startswith(".")]  # staging cleaned

    assert [d["name"] for d in dclient.get("/datasets").json()] == ["t"]
    info = dclient.post("/datasets/t/session").json()
    assert info["dataset"] == "t" and info["n_frames"] == N
    result = dclient.post(f"/sessions/{info['session_id']}/convert", json={"frames": "3"}).json()
    assert dclient.get(f"/sessions/{info['session_id']}/results/{result['result_id']}").status_code == 200

    # Closing the session removes only its results, never the stored data.
    dclient.delete(f"/sessions/{info['session_id']}")
    assert (tmp_path / "data" / "t" / "t_data_000001.h5").exists()


def test_dataset_name_conflict_overwrite_and_delete(dclient, tmp_path):
    assert dclient.post("/datasets", files=src_files(tmp_path), data={"name": "A25"}).status_code == 200
    assert dclient.post("/datasets", files=src_files(tmp_path), data={"name": "A25"}).status_code == 409
    info = dclient.post("/datasets/A25/session").json()
    ok = dclient.post("/datasets", files=src_files(tmp_path), data={"name": "A25", "overwrite": "true"})
    assert ok.status_code == 200
    assert dclient.get(f"/sessions/{info['session_id']}").status_code == 404  # stale session dropped
    assert dclient.delete("/datasets/A25").status_code == 200
    assert dclient.get("/datasets").json() == []
    assert dclient.post("/datasets/A25/session").status_code == 404


def test_dataset_rejects_bad_input(dclient, tmp_path):
    assert dclient.post("/datasets", files=src_files(tmp_path, ["t_master.h5"])).status_code == 400  # data file missing
    assert dclient.post("/datasets", files=src_files(tmp_path, ["t_data_000001.h5"])).status_code == 400  # no master
    assert dclient.post("/datasets", files=src_files(tmp_path), data={"name": "../x"}).status_code == 400
    assert dclient.delete("/datasets/..").status_code in (400, 404)
    assert dclient.get("/datasets").json() == []
    assert list((tmp_path / "data").iterdir()) == []  # nothing left behind
