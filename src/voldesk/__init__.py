"""voldesk: a cross-asset volatility research and trading desk.

Subpackages
-----------
pricing     Black-Scholes/Black-76 prices, Greeks, implied volatility, American lattices
data        WRDS pulls with provenance, the parquet store, rates/dividends/calendars, quality checks, synthetic fallback
surface     smile/surface models (SVI, SSVI, eSSVI, SABR, Heston, local vol, LSV, rough Bergomi) and arbitrage checks
strategies  option structures, Greeks, P&L attribution, costs and margin
backtest    daily mark-to-market engine on historical chains, delta hedging
risk        portfolio aggregation, scenario grids, historical stress, VaR / ES
forecast    realized and implied volatility, vol of vol, correlation and regime forecasts with walk-forward evaluation
dispersion  implied correlation and dispersion trades
web         export of the interactive demo's data bundle
"""
__version__ = "0.1.0"
