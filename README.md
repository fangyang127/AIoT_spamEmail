# Spam/Ham Classifier

這個專案實作從資料載入、前處理、模型訓練到互動式視覺化的完整流程。專案使用 scikit-learn（TF-IDF + LogisticRegression）訓練一個簡單的二分類文字模型，並使用 Streamlit 展示預測、分析與評估結果。

**Source Reference:** https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity.git

**Demo Site:** https://aiotspamemail-fang.streamlit.app/

## 主要功能

- 可重現的訓練流程（`src/train.py`）會輸出模型與評估檔到 `models/`
- Streamlit Dashboard（`app/streamlit_app.py`）包含：
	- 文字輸入與即時預測
	- 範例按鈕（spam / ham）與自動填入
	- 顯示 spam 機率（數值與條狀圖）及彩色化結果
	- 模型性能視覺化（混淆矩陣、ROC、Precision-Recall、threshold sweep）
	- Top tokens by class（TF-IDF 重要字詞）
- OpenSpec 目錄包含專案能力與變更提案示例，方便實作規格驅動開發

## 先決條件

- Python 3.8+（建議 3.10/3.11）
- 建議在 virtual environment 中運行

## 快速開始（Windows PowerShell）

1) 建立並啟用虛擬環境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) 安裝相依套件

```powershell
pip install -r requirements.txt
```

3) 執行訓練（範例）

```powershell
python -m src.train
```

執行後會在 `models/` 產生：

- `model--logistic--v0.1.joblib`
- `metrics.json`
- （選項）混淆矩陣與 ROC 圖檔

4) 啟動 Streamlit Dashboard

```powershell
streamlit run app/streamlit_app.py
```

5) 執行測試

```powershell
pytest -q
```

## 專案結構（重點）

- `app/` — `streamlit_app.py`：Streamlit 應用，UI 與互動邏輯
- `data/` — 範例資料（CSV）
- `models/` — 訓練後的模型與評估 artifact
- `src/` — 核心程式碼
	- `data_loader.py` — 下載/讀取資料與簡單清理
	- `preprocessing.py` — TF-IDF 向量器建構
	- `model.py` — pipeline 建構與儲存/載入工具
	- `train.py` — 訓練流程、評估並儲存 artifact
	- `evaluate.py` — 評估與圖表工具
- `openspec/` — capability 規格與變更提案範例
- `tests/` — pytest 測試

## 訓練與模型細節

- Pipeline：TF-IDF vectorizer → LogisticRegression
- 預處理：小寫化、停用詞（英語）、TF-IDF 篩選參數由 `src/preprocessing.py` 控制
- 模型輸出：使用 `joblib.dump` 儲存整個 pipeline，方便在 Streamlit 中直接載入與 predict

若想紀錄更多訓練超參數或 reproducibility 信息，可在 `src/train.py` 中擴充 metadata 寫入（例如 sklearn 版本、訓練日期、資料切分資訊）。

## 評估

- `src/train.py` 會計算並儲存主要指標（accuracy, precision, recall, f1, roc_auc）到 `models/metrics.json`
- Dashboard 提供混淆矩陣、ROC 與 Precision-Recall 圖表，並支援 threshold sweep 檢視不同閾值的 precision/recall/f1

## 效能與延伸（建議）

- 如果 Dashboard 啟動很慢，考慮：
	- 在訓練階段預先計算 Top tokens 與 threshold 表格並儲存在 `models/`，於啟動時直接載入
	- 使用 `st.cache_data` / `st.cache_resource` 快取昂貴計算
- 若想進一步提升模型：嘗試更複雜的特徵（ngram、詞性過濾）、其他分類器（SVM、RandomForest）或微調 class weight

## CI 與 OpenSpec

- 本 repo 含有 GitHub Actions（`.github/workflows/`）— CI 會在 push/PR 執行 pytest，並可選地驗證 `openspec/changes/` 的變更
- `package.json` 內有 openspec 的 devDependency，方便在 CI 或本機使用相同版本的驗證工具

## 常見問題（Troubleshooting）

- Streamlit 在部署時找不到模型或 data：
	- 確認 `models/` 與必要的資料已包含在部署環境（或在啟動前執行 `python -m src.train`）
- 部署後看到空白頁面或匯入錯誤：
	- 檢查 `streamlit_import_error.log`（如果專案中有寫入）或查看 Streamlit Cloud 的部署日誌

