# Spam/Ham Classifier

一個簡潔的教學專案：使用 scikit-learn 訓練一個簡易的 Spam/Ham（二分類）文字分類器，並以 Streamlit 提供互動式展示與分析介面。

主要特性
- 以 TF-IDF + LogisticRegression 建立的 scikit-learn Pipeline
- 訓練腳本會輸出模型（`.joblib`）、評估指標（`metrics.json`）與圖表到 `models/`
- Streamlit Dashboard（`app/streamlit_app.py`）：即時預測、範例填入、機率顯示、混淆矩陣、ROC/PR 與 threshold sweep
- OpenSpec 資料夾（`openspec/`）包含專案規格與變更提案範例

快速開始（Windows PowerShell）

1) 建議建立並啟用虛擬環境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) 安裝相依套件

```powershell
pip install -r requirements.txt
```

3) 執行訓練（會下載 dataset 並在 `models/` 產生 artifact）

```powershell
python -m src.train
```

4) 啟動 Dashboard（訓練後或直接開啟查看示範）

```powershell
streamlit run app/streamlit_app.py
```

5) 執行測試

```powershell
pytest -q
```

開發者說明
- 程式主要放在 `src/`（data loading, preprocessing, model, training, evaluate）
- Dashboard 實作在 `app/streamlit_app.py`，包含範例按鈕、可調 threshold、Top tokens 檢視與 performance plots
- 若部署到 Streamlit Cloud，請確認 `models/` 中有訓練好的 artifact（或在啟動前執行 train）

OpenSpec 與 CI
- `openspec/` 用來管理 capability 規格與 change proposals（參考專案內範例檔案）
- CI（`.github/workflows/`）會在 push/PR 時執行 pytest 並驗證 openspec 變更

檔案一覽
- `app/streamlit_app.py` — Streamlit 應用
- `src/` — 核心程式碼（data_loader, preprocessing, model, train, evaluate）
- `models/` — 訓練後的模型與評估 artifact
- `data/` — 範例資料（CSV）

其他
- 若要我幫忙：把 README 翻成英文、加入範例截圖、或自動把 token 預計算到訓練 artifact，我都可以代為實作。

授權
- 目前未指定授權；如需我可協助加上 MIT / Apache-2.0 等授權檔並提交 PR。
