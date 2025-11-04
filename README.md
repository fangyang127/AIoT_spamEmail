# AIoT — Spam SMS Classifier

這是一個課堂專案（IoT Data Analysis and Applications / homework 3）：使用 Python 與 scikit-learn 建置一個可重現的垃圾簡訊（SMS）分類器，並以 Streamlit 提供互動式展示。

主要功能
- 資料載入與前處理（`src/data_loader.py`, `src/preprocessing.py`）
- 使用 Logistic Regression 的 scikit-learn pipeline（`src/model.py`）
- 訓練與評估流程（`src/train.py`, `src/evaluate.py`），會輸出模型與評估報告到 `models/`
- Streamlit Dashboard（`app/streamlit_app.py`）: 預測、動態 threshold、機率分佈、混淆矩陣與 ROC 圖
- OpenSpec 規格與提案（`openspec/`）以支援規格驅動開發

快速開始（PowerShell）

1) 建議建立並啟用虛擬環境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) 安裝 Python 套件

```powershell
pip install -r requirements.txt
```

3) 執行訓練（會自動下載 dataset 並在 `models/` 產生 artifact）

```powershell
python -m src.train
```

4) 啟動 Streamlit Dashboard（在訓練後可以立即查看模型與圖表）

```powershell
streamlit run app/streamlit_app.py
```

5) 執行測試

```powershell
pytest -q
```

OpenSpec / 提案流程
- 專案包含 `openspec/` 目錄：
  - `openspec/project.md`：專案背景與規範
  - `openspec/specs/`：已部署/真實的 capability 規格
  - `openspec/changes/`：待審查的變更提案（每個提案包含 `proposal.md`, `tasks.md`, 以及 delta specs）
  - `openspec/proposals/`：草案位置（可選）
- 在建立或修改 capability 時，請先建立 change proposal 並使用 OpenSpec CLI 驗證：

```powershell
npx @fission-ai/openspec validate <change-id> --strict
```

CI（GitHub Actions）
- 本專案包含 `.github/workflows/ci.yml`：
  - 在 `push` / `pull_request` 到 `main` 時執行
  - 執行單元測試（`pytest`）並嘗試對變更過的 `openspec/changes/<id>/` 執行 `openspec validate`。
- 我們也提供 `package.json` 並將 `@fission-ai/openspec` 加入 `devDependencies`，方便在 CI 與本機一致地驗證規格。

開發與貢獻
- 分支策略：
  - `main`：穩定 / 已部署
  - feature branches：`feat/<short-desc>`，PR 目標設為 `main` 或 `dev`（視團隊流程）
- 提交訊息建議採用 Conventional Commits

資料與隱私
- `data/sms_spam_no_header.csv` 僅用於課程示範；請勿將真實含 PII 的資料上傳到公開 repository。

檔案一覽（重點）
- `src/`：程式碼（data_loader, preprocessing, model, train, evaluate）
- `app/streamlit_app.py`：Dashboard
- `models/`：輸出的 model artifact（`.joblib`）與 `metrics.json`, 圖檔
- `openspec/`：規格與提案
- `requirements.txt`, `package.json`, `package-lock.json`

聯絡與後續
- 若要我幫你：
  - 將 README 翻成英文或加入範例截圖
  - 把 Dashboard 部署到 Streamlit Cloud
  - 擴充 CI（例如加入 `npm ci`、或執行小型 smoke training）

License
- 預設未指定；若需要我可以幫你加入常見授權（MIT / Apache-2.0 / CC-BY）並提交修改。
