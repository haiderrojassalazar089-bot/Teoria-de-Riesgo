"""
backend/app/sp500_service.py
Lista del S&P 500 con nombres y sectores.
Intenta Wikipedia primero; fallback robusto con 80 empresas si falla.
"""
from __future__ import annotations
import logging
import time
import pandas as pd

logger = logging.getLogger(__name__)

_sp500_cache: dict = {"data": [], "timestamp": 0.0}
_CACHE_TTL = 86400  # 24 horas

# ── Fallback robusto con nombres y sectores ───────────────────
_FALLBACK: list[dict] = [
    # Tecnología
    {"ticker": "AAPL",  "name": "Apple Inc.",                  "sector": "Information Technology"},
    {"ticker": "MSFT",  "name": "Microsoft Corporation",        "sector": "Information Technology"},
    {"ticker": "NVDA",  "name": "NVIDIA Corporation",           "sector": "Information Technology"},
    {"ticker": "GOOGL", "name": "Alphabet Inc. (Class A)",      "sector": "Communication Services"},
    {"ticker": "META",  "name": "Meta Platforms Inc.",          "sector": "Communication Services"},
    {"ticker": "TSLA",  "name": "Tesla Inc.",                   "sector": "Consumer Discretionary"},
    {"ticker": "AVGO",  "name": "Broadcom Inc.",                "sector": "Information Technology"},
    {"ticker": "AMD",   "name": "Advanced Micro Devices",       "sector": "Information Technology"},
    {"ticker": "INTC",  "name": "Intel Corporation",            "sector": "Information Technology"},
    {"ticker": "QCOM",  "name": "Qualcomm Inc.",                "sector": "Information Technology"},
    {"ticker": "TXN",   "name": "Texas Instruments",            "sector": "Information Technology"},
    {"ticker": "ADBE",  "name": "Adobe Inc.",                   "sector": "Information Technology"},
    {"ticker": "CRM",   "name": "Salesforce Inc.",              "sector": "Information Technology"},
    {"ticker": "INTU",  "name": "Intuit Inc.",                  "sector": "Information Technology"},
    {"ticker": "IBM",   "name": "IBM Corporation",              "sector": "Information Technology"},
    {"ticker": "ADI",   "name": "Analog Devices Inc.",          "sector": "Information Technology"},
    {"ticker": "LRCX",  "name": "Lam Research Corporation",     "sector": "Information Technology"},
    {"ticker": "NOW",   "name": "ServiceNow Inc.",              "sector": "Information Technology"},
    {"ticker": "PANW",  "name": "Palo Alto Networks",           "sector": "Information Technology"},
    {"ticker": "ORCL",  "name": "Oracle Corporation",           "sector": "Information Technology"},
    # Financiero
    {"ticker": "JPM",   "name": "JPMorgan Chase & Co.",         "sector": "Financials"},
    {"ticker": "BAC",   "name": "Bank of America Corp.",        "sector": "Financials"},
    {"ticker": "GS",    "name": "Goldman Sachs Group",          "sector": "Financials"},
    {"ticker": "MS",    "name": "Morgan Stanley",               "sector": "Financials"},
    {"ticker": "V",     "name": "Visa Inc.",                    "sector": "Financials"},
    {"ticker": "MA",    "name": "Mastercard Inc.",              "sector": "Financials"},
    {"ticker": "WFC",   "name": "Wells Fargo & Company",        "sector": "Financials"},
    {"ticker": "AXP",   "name": "American Express Company",     "sector": "Financials"},
    {"ticker": "BLK",   "name": "BlackRock Inc.",               "sector": "Financials"},
    {"ticker": "SPGI",  "name": "S&P Global Inc.",              "sector": "Financials"},
    {"ticker": "CB",    "name": "Chubb Limited",                "sector": "Financials"},
    {"ticker": "AON",   "name": "Aon plc",                      "sector": "Financials"},
    {"ticker": "MMC",   "name": "Marsh & McLennan Companies",   "sector": "Financials"},
    # Salud
    {"ticker": "JNJ",   "name": "Johnson & Johnson",            "sector": "Health Care"},
    {"ticker": "UNH",   "name": "UnitedHealth Group",           "sector": "Health Care"},
    {"ticker": "PFE",   "name": "Pfizer Inc.",                  "sector": "Health Care"},
    {"ticker": "ABBV",  "name": "AbbVie Inc.",                  "sector": "Health Care"},
    {"ticker": "MRK",   "name": "Merck & Co. Inc.",             "sector": "Health Care"},
    {"ticker": "TMO",   "name": "Thermo Fisher Scientific",     "sector": "Health Care"},
    {"ticker": "DHR",   "name": "Danaher Corporation",          "sector": "Health Care"},
    {"ticker": "AMGN",  "name": "Amgen Inc.",                   "sector": "Health Care"},
    {"ticker": "GILD",  "name": "Gilead Sciences Inc.",         "sector": "Health Care"},
    {"ticker": "REGN",  "name": "Regeneron Pharmaceuticals",    "sector": "Health Care"},
    {"ticker": "VRTX",  "name": "Vertex Pharmaceuticals",       "sector": "Health Care"},
    {"ticker": "ISRG",  "name": "Intuitive Surgical Inc.",      "sector": "Health Care"},
    {"ticker": "SYK",   "name": "Stryker Corporation",          "sector": "Health Care"},
    {"ticker": "BSX",   "name": "Boston Scientific Corporation", "sector": "Health Care"},
    {"ticker": "MDT",   "name": "Medtronic plc",                "sector": "Health Care"},
    {"ticker": "ELV",   "name": "Elevance Health Inc.",         "sector": "Health Care"},
    {"ticker": "CI",    "name": "Cigna Group",                  "sector": "Health Care"},
    # Energía
    {"ticker": "XOM",   "name": "ExxonMobil Corporation",       "sector": "Energy"},
    {"ticker": "CVX",   "name": "Chevron Corporation",          "sector": "Energy"},
    {"ticker": "COP",   "name": "ConocoPhillips",               "sector": "Energy"},
    {"ticker": "SLB",   "name": "Schlumberger Limited",         "sector": "Energy"},
    {"ticker": "EOG",   "name": "EOG Resources Inc.",           "sector": "Energy"},
    {"ticker": "PXD",   "name": "Pioneer Natural Resources",    "sector": "Energy"},
    {"ticker": "MPC",   "name": "Marathon Petroleum Corp.",     "sector": "Energy"},
    {"ticker": "PSX",   "name": "Phillips 66",                  "sector": "Energy"},
    # Consumo discrecional
    {"ticker": "AMZN",  "name": "Amazon.com Inc.",              "sector": "Consumer Discretionary"},
    {"ticker": "HD",    "name": "Home Depot Inc.",              "sector": "Consumer Discretionary"},
    {"ticker": "MCD",   "name": "McDonald's Corporation",       "sector": "Consumer Discretionary"},
    {"ticker": "NKE",   "name": "Nike Inc.",                    "sector": "Consumer Discretionary"},
    {"ticker": "SBUX",  "name": "Starbucks Corporation",        "sector": "Consumer Discretionary"},
    {"ticker": "LOW",   "name": "Lowe's Companies Inc.",        "sector": "Consumer Discretionary"},
    {"ticker": "TGT",   "name": "Target Corporation",           "sector": "Consumer Discretionary"},
    {"ticker": "BKNG",  "name": "Booking Holdings Inc.",        "sector": "Consumer Discretionary"},
    # Consumo básico
    {"ticker": "WMT",   "name": "Walmart Inc.",                 "sector": "Consumer Staples"},
    {"ticker": "PG",    "name": "Procter & Gamble Company",     "sector": "Consumer Staples"},
    {"ticker": "KO",    "name": "Coca-Cola Company",            "sector": "Consumer Staples"},
    {"ticker": "PEP",   "name": "PepsiCo Inc.",                 "sector": "Consumer Staples"},
    {"ticker": "COST",  "name": "Costco Wholesale Corporation", "sector": "Consumer Staples"},
    {"ticker": "PM",    "name": "Philip Morris International",  "sector": "Consumer Staples"},
    {"ticker": "MO",    "name": "Altria Group Inc.",            "sector": "Consumer Staples"},
    {"ticker": "CL",    "name": "Colgate-Palmolive Company",    "sector": "Consumer Staples"},
    # Industriales
    {"ticker": "GE",    "name": "GE Aerospace",                 "sector": "Industrials"},
    {"ticker": "HON",   "name": "Honeywell International",      "sector": "Industrials"},
    {"ticker": "CAT",   "name": "Caterpillar Inc.",             "sector": "Industrials"},
    {"ticker": "RTX",   "name": "RTX Corporation",              "sector": "Industrials"},
    {"ticker": "UPS",   "name": "United Parcel Service",        "sector": "Industrials"},
    {"ticker": "BA",    "name": "Boeing Company",               "sector": "Industrials"},
    {"ticker": "LMT",   "name": "Lockheed Martin Corporation",  "sector": "Industrials"},
    {"ticker": "ITW",   "name": "Illinois Tool Works",          "sector": "Industrials"},
    {"ticker": "DE",    "name": "Deere & Company",              "sector": "Industrials"},
    # Telecomunicaciones
    {"ticker": "NFLX",  "name": "Netflix Inc.",                 "sector": "Communication Services"},
    {"ticker": "DIS",   "name": "Walt Disney Company",          "sector": "Communication Services"},
    {"ticker": "CMCSA", "name": "Comcast Corporation",          "sector": "Communication Services"},
    {"ticker": "T",     "name": "AT&T Inc.",                    "sector": "Communication Services"},
    {"ticker": "VZ",    "name": "Verizon Communications",       "sector": "Communication Services"},
    # Utilities
    {"ticker": "NEE",   "name": "NextEra Energy Inc.",          "sector": "Utilities"},
    {"ticker": "SO",    "name": "Southern Company",             "sector": "Utilities"},
    {"ticker": "DUK",   "name": "Duke Energy Corporation",      "sector": "Utilities"},
    # Real Estate
    {"ticker": "AMT",   "name": "American Tower Corporation",   "sector": "Real Estate"},
    {"ticker": "PLD",   "name": "Prologis Inc.",                "sector": "Real Estate"},
    {"ticker": "EQIX",  "name": "Equinix Inc.",                 "sector": "Real Estate"},
    {"ticker": "PSA",   "name": "Public Storage",               "sector": "Real Estate"},
    # Materiales
    {"ticker": "LIN",   "name": "Linde plc",                    "sector": "Materials"},
    {"ticker": "APD",   "name": "Air Products and Chemicals",   "sector": "Materials"},
    {"ticker": "SHW",   "name": "Sherwin-Williams Company",     "sector": "Materials"},
    {"ticker": "FCX",   "name": "Freeport-McMoRan Inc.",        "sector": "Materials"},
    # BRK
    {"ticker": "BRK-B", "name": "Berkshire Hathaway Inc.",      "sector": "Financials"},
    {"ticker": "ACN",   "name": "Accenture plc",                "sector": "Information Technology"},
    {"ticker": "ZTS",   "name": "Zoetis Inc.",                  "sector": "Health Care"},
    {"ticker": "MMM",   "name": "3M Company",                   "sector": "Industrials"},
]


def get_sp500_info() -> list[dict]:
    """
    Retorna lista con ticker + nombre + sector del S&P 500.
    Intenta Wikipedia primero; usa fallback robusto si falla.
    """
    now = time.time()
    if _sp500_cache["data"] and (now - _sp500_cache["timestamp"]) < _CACHE_TTL:
        return _sp500_cache["data"]

    try:
        logger.info("Descargando lista S&P 500 desde Wikipedia...")
        table = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            header=0,
        )[0]
        result = []
        for _, row in table.iterrows():
            ticker = str(row["Symbol"]).strip().replace(".", "-")
            name   = str(row.get("Security", row.get("Company", ticker))).strip()
            sector = str(row.get("GICS Sector", "")).strip()
            result.append({"ticker": ticker, "name": name, "sector": sector})

        if result:
            _sp500_cache["data"] = result
            _sp500_cache["timestamp"] = now
            logger.info(f"S&P 500: {len(result)} empresas cargadas desde Wikipedia.")
            return result
    except Exception as e:
        logger.warning(f"Wikipedia no disponible: {e}. Usando fallback.")

    # Fallback con nombres y sectores reales
    _sp500_cache["data"] = _FALLBACK
    _sp500_cache["timestamp"] = now
    logger.info(f"Fallback activado: {len(_FALLBACK)} empresas.")
    return _FALLBACK


def get_sp500_tickers() -> list[str]:
    """Retorna solo los tickers del S&P 500."""
    return [i["ticker"] for i in get_sp500_info()]


def is_sp500_ticker(ticker: str) -> bool:
    """Verifica si un ticker pertenece al S&P 500."""
    valid = {t.upper() for t in get_sp500_tickers()}
    return ticker.upper().replace(".", "-") in valid


def validate_tickers_sp500(tickers: list[str]) -> tuple[bool, list[str]]:
    """
    Valida que todos los tickers estén en el S&P 500.
    Retorna (ok, lista_de_invalidos).
    """
    valid = {t.upper() for t in get_sp500_tickers()}
    invalid = [t for t in tickers if t.upper().replace(".", "-") not in valid]
    return len(invalid) == 0, invalid