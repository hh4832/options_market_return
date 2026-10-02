# 研究與開發規範

## Reproducibility
- GitHub 是 single source of truth。核心邏輯放 `src/*.py`；Notebook 只設定環境、執行、驗證與展示。
- 支援 GitHub → Colab → Run All；使用 pathlib 與相對路徑，不依賴固定電腦。
- 目標 Python 3.11；環境隔離，不修改系統 Python。新增依賴須同步 requirements.txt。
- 每次正式 run 記錄完整 Git commit、branch、dirty 狀態、套件版本、資料範圍與 config。未 commit 或 dirty working tree 不可作正式研究。
- Asia/Taipei；archive `YYYYMMDD_HHMMSS_<git_commit>`；禁止覆蓋舊 outputs、資料或 archive。
- Drive 根目錄 `/content/drive/MyDrive/00Quant_Research/options_market_return`。
- 不提交 token、憑證、.env、原始授權資料與產生型 outputs。Secret 只從環境變數／Colab Secret 取得。
- 修改前確認 repository、remote、branch、status、Python 與 dependencies，保留既有變更。
- commit／push／pull／merge／rebase／reset 等須有使用者明確授權；禁止 force push。

## Research integrity
- 先說明市場機制與假設，再定義規則。Put/Call 不預先指定 bullish 或 bearish。
- 每次檢查 look-ahead、survivorship、selection、data snooping、overlapping-return dependence、missing data、duplicate/malformed dates、non-finite values、signal/outcome alignment、樣本不足、few-year concentration、parameter island、regime dependence。
- 禁止 silent fill、missing return=0、因結果不好改 tests、事後挑漂亮參數定義假設、用單一最佳 cell 宣稱有效。
- 一次只調整一個核心研究變數；說明目的、預期影響、副作用。策略驗證另須交易成本、滑價、流動性與 tail risk。
- VOLUME 與 OPEN_INTEREST 分開；第一版只使用 tw_option_put_call_ratio。0050 僅用確認過的還原開／收盤；無法確認就停止，不 fallback raw price。
- inference 必須保留交易日間距使用 HAC；two-sided。zero 與 unconditional comparison 的 Family／Global BH-FDR 分開。

## Diagnostics 是正式產物
- 成功與失敗均保留資料品質 diagnostics；不可 silent failure。
- 至少：data coverage/date range/missing/duplicate/malformed/non-finite、0050 alignment、signal-entry-outcome dates、bin counts、rolling/horizon usable counts、annual counts、bucket coverage、signal distribution、extreme-bin n、FDR tests/discoveries、HAC failures、neighbor consistency、year concentration。
- 正式輸出 diagnostics/ 包含 data_coverage、missing_data_report、date_alignment、signal_diagnostics、percentile_bucket_counts、rolling_window_coverage、outcome_coverage、hac_diagnostics、fdr_diagnostics、annual_contribution、parameter_neighbor_diagnostics CSV 與 diagnostics_summary.md。

## Observations 是正式產物
- observations/observation_summary.csv 與 .md 必須由實際結果生成；與 conclusions 分開。
- 描述單調性（不可等同線性）、extreme-only、rolling/horizon 方向、年度集中、parameter island、Family-only discoveries、極端樣本不足。
- 不得宣稱有效策略、alpha、因果或最好訊號。PCR 與 CallShare 不是獨立證據。
- evidence 用客觀 flags，禁止主觀 score；Global FDR 通過也不等同策略有效。
- 主動列反對者觀點與尚未證實限制。最終研究決策僅「保留／修改後再測／淘汰」；資料不足時標示無法判定。

## Validation 與交付
- 測試公式、zero denominator、percentile boundaries/no look-ahead、O1→Cn／交易日、adjusted validation、HAC／FDR universes、年度／neighbors／missing。
- 先找失敗根因，不改 tests 迎合結果；合成資料不能當成真實市場績效。
- 回報檔案、公式、keys、tests、未驗證項目、Colab、archive、Git diff/status；「修改／測試／commit／push」分開說明。
