"""The S-REIT universe, with sub-sector labels.

Singapore is one of the largest REIT markets in Asia and the sector behaves
very differently from the US equities most retail quant work is built on: it
is yield-driven, heavily levered, and unusually sensitive to interest rates.
Those are the properties this repo is built around.

On the sub-sector labels
------------------------
They are curated here rather than taken from the data vendor, because the
vendor's are wrong in ways that matter for this analysis. Yahoo Finance
classifies **Keppel DC REIT as "REIT - Office"**; it is a pure data-centre
REIT, whose lease structure, tenant base and rate sensitivity have nothing in
common with an office landlord. It also files CapitaLand Integrated Commercial
Trust under "Retail" when it is roughly half office by asset value.

Sub-sector is the main axis every comparison in this repo cuts along, so
inheriting those errors would corrupt the results silently. Where a
classification is genuinely arguable, the label here says so.
"""

from __future__ import annotations

import pandas as pd

# ticker -> (name, sub_sector, note)
SREITS: dict[str, tuple[str, str, str]] = {
    "C38U.SI": ("CapitaLand Integrated Commercial Trust", "Diversified",
                "retail + office; Yahoo files it as pure Retail"),
    "A17U.SI": ("CapitaLand Ascendas REIT", "Industrial", ""),
    "M44U.SI": ("Mapletree Logistics Trust", "Industrial", "logistics"),
    "ME8U.SI": ("Mapletree Industrial Trust", "Industrial", "growing data-centre weight"),
    "BUOU.SI": ("Frasers Logistics & Commercial Trust", "Industrial", "logistics + some commercial"),
    "M1GU.SI": ("Alpha Integrated REIT", "Industrial", "formerly Sabana"),
    "AJBU.SI": ("Keppel DC REIT", "Data Centre",
                "Yahoo mislabels this as Office -- it is a pure data-centre REIT"),
    "N2IU.SI": ("Mapletree Pan Asia Commercial Trust", "Office", ""),
    "K71U.SI": ("Keppel REIT", "Office", ""),
    "UD1U.SI": ("IREIT Global", "Office", "European assets"),
    "TS0U.SI": ("OUE REIT", "Office", ""),
    "OXMU.SI": ("Prime US REIT", "Office", "US assets, SGD-listed"),
    "CMOU.SI": ("KORE US REIT", "Office", "US assets, SGD-listed"),
    "J69U.SI": ("Frasers Centrepoint Trust", "Retail", "suburban malls"),
    "P40U.SI": ("Starhill Global REIT", "Retail", ""),
    "CRPU.SI": ("Sasseur REIT", "Retail", "China outlet malls"),
    "AU8U.SI": ("CapitaLand China Trust", "Retail", "China assets"),
    "HMN.SI": ("CapitaLand Ascott Trust", "Hospitality", ""),
    "Q5T.SI": ("Far East Hospitality Trust", "Hospitality", ""),
    "JYEU.SI": ("Lendlease Global Commercial REIT", "Diversified", ""),
    "C2PU.SI": ("Parkway Life REIT", "Healthcare", "long WALE, CPI-linked rents"),
    "CY6U.SI": ("CapitaLand India Trust", "Diversified",
                "a business trust rather than a REIT proper; kept, flagged"),
}

# ABF Singapore Bond Index Fund -- an SGX-listed fund holding Singapore
# government and quasi-government bonds. Used as the SG interest-rate proxy.
# See src/rates.py for why this stands in for the SGS 10-year yield and what
# that substitution costs.
RATE_PROXY = "A35.SI"
# Straits Times Index ETF -- the local equity benchmark.
MARKET_PROXY = "ES3.SI"


def tickers() -> list[str]:
    return sorted(SREITS)


def universe_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": t, "name": name, "sub_sector": sector, "note": note}
            for t, (name, sector, note) in sorted(SREITS.items())
        ]
    )


def sub_sectors() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ticker, (_, sector, _) in sorted(SREITS.items()):
        out.setdefault(sector, []).append(ticker)
    return out


def sub_sector_of(ticker: str) -> str | None:
    entry = SREITS.get(ticker)
    return entry[1] if entry else None
