# data/regime/

Place `daily_regime_labels.csv` here before running training.

Generate by running:
    jupyter notebook notebooks/00_market_regime_explorer_v5.ipynb

The notebook downloads 7,666 days of daily OHLCV for 9 instruments,
runs GMM regime classification, KMeans consensus scoring, volatility
detection, and exports the CSV with 35 columns.

Required columns consumed by the pipeline:
    gmm2_state, km_label_63d, vol_regime,
    regime_quality_norm, gs_quartile_enc, cu_au_regime_enc
