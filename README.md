# 台灣選擇權與 0050 後續報酬

研究成交活動及留倉結構是否具有單調、極端、反轉或延續關係；不預先指定多空方向。這是探索研究，不能以單一最佳 cell 宣稱有效。

## 執行
開啟 `notebooks/run_options_market_return.ipynb` 至 Colab，Run All。支援 private repository 的 `GITHUB_TOKEN` Secret；FinLab 使用 `FINLAB_API_TOKEN` Secret 或環境變數。不得將憑證寫入檔案。正式研究需 committed、clean checkout 與 Python 3.11；本次開發不 commit/push。

缺少 FinLab credential 標記 `DATA_VALIDATION_BLOCKED_BY_CREDENTIAL`，不能解讀為資料品質或 key 驗證失敗。live validation 必須於有 credential 的環境完成，失敗即停止，沒有 raw-price fallback。

## 定義
資料僅 `tw_option_put_call_ratio`，VOLUME、OPEN_INTEREST 分開。各 source 的 PUT、CALL、PCR=Put/Call、CALLSHARE=Call/(Put+Call) 各有 level 與 1/3/5 交易日差值，共 32 signals。差值為 x[t]-x[t-k]，不是百分比；分母零保留 NaN。供應商百分比除以 100 後交叉驗證。

Rolling 60/120/252/504/756，當期 rank=100*(小於當期數+0.5*等於當期數)/window，僅用截至 t 完整窗口。Bins=[0,5)、[5,20)、[20,40)、[40,60)、[60,80)、[80,95]、(95,100]。

Outcome=adjusted close[t+h]/adjusted open[t+1]-1，h=1/2/3/5/10/20，使用完整股票交易日序列；缺價不壓縮日曆、不填補。keys 為 `etl:adj_open`、`etl:adj_close`；live loader 驗證 available keys、0050 欄位，validation 驗證日期、正價格及 adjusted/raw open-close factor 一致性。raw keys `price:開盤價`、`price:收盤價` 僅交叉驗證，不用作 outcome。共同歷史不足停止。

HAC 為完整交易日網格上的 ratio-of-means influence contrast，Bartlett/Newey-West lag=h，有限樣本 T/(T-1)，雙尾 normal Wald。未選日期為零 influence，不是零 return；對照樣本重疊協方差保留。zero 與 excess 各自 BH：Family=source×signal definition（預設 210 planned cells），Global=32 families（6720 planned cells）；NaN tests 排除並回報 valid/planned counts。extreme 五種預先定義 contrasts 各有獨立 universe。

七 bins Spearman 為描述性單調性，不是線性、也不是正式時間序列 HAC trend 推論。年度依 signal year 分組，contribution=abs(n_year*annual_excess)/總和。鄰近各軸取前後一格；same sign、0.5–2倍 effect 與 Family significance 記錄 robustness；孤立顯著標記 PARAMETER_ISLAND。A/B/C/D 只代表 evidence；Global 有警訊另標 GLOBAL_WITH_WARNINGS。

## 輸出與驗證
`outputs/YYYYMMDD_HHMMSS_<git_commit>/`，Asia/Taipei，禁止覆蓋；Drive `/content/drive/MyDrive/00Quant_Research/options_market_return`。包含 research_results、annual_results、annual_summary、monotonicity_results、extreme_results、config、metadata、run_info、reproducibility_data、figures、diagnostics、observations。

Diagnostics：data_coverage、missing_data_report、date_alignment、signal_diagnostics、percentile_bucket_counts、rolling_window_coverage、outcome_coverage、hac_diagnostics、fdr_diagnostics、annual_contribution、parameter_neighbor_diagnostics、ratio_cross_validation、zero_denominators、annual_sample_counts 與 summary。Observations CSV/MD 由結果計算，涵蓋單調性、extremes、HAC/FDR、年度集中與穩定、window/horizon 方向、parameter island。

離線測試：`.venv/bin/python -m pytest -q`。合成資料只能驗證程式；FinLab live、Colab Run All、Drive mount 與真實績效必須另外驗證。

## 反對者觀點
Put/Call OI 可能包含 protective put、short put、covered call、spread、market-maker hedging、expiry effects、結構改變，不能直接代表 bearish/bullish。PCR 與 CallShare 高度相關；windows/bins/horizons 亦相關。BH 不能完全排除 data snooping。0050 單一存續 ETF 與起始日期有 selection/survivorship 限制；尚無樣本外、成本、滑價、流動性驗證，無法判定可交易性。年度年度集中可能代表 regime dependence。最終決策僅保留／修改後再測／淘汰，無市場資料時無法判定。
