# Changelog

## 4.9.0

- Require `mhctools>=3.5.0`.
- Rename CLI sorting flags to `--sort-by` and `--sort-direction`.
- Add Python API `sort_by=` and `.sort_by(...)`, while keeping `rank_by` as a compatibility alias.
- Treat comma-separated `--sort-by` keys as lexicographic tie breakers, with fallthrough on missing values.
- Document upstream `mhctools 3.5.0` support for multi-predictor CLI invocations and the updated NetChop/Pepsickle behavior.
