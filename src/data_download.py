from __future__ import annotations

from eodhd import APIClient
import pandas as pd
import os


def download_eod_data(
    symbol: str,
    start_date: str,
    end_date: str,
    output_path: str,
) -> None:
    api_key = os.getenv("EODHD_API")
    if api_key is None:
        raise ValueError("Brak klucza API EODHD")

    api = APIClient(api_key)

    resp = api.get_eod_historical_stock_market_data(
        symbol=symbol,
        period="d",
        from_date=start_date,
        to_date=end_date,
        order="a",
    )

    if resp is None:
        raise ValueError("Nie można pobrać danych, sprawdź parametry")

    df = pd.DataFrame(resp)

    df.to_csv(output_path, index=False)
    print(f"Zapisano do: {output_path}")


if __name__ == "__main__":
    download_eod_data(
        symbol="SPY.US",
        start_date="2010-01-01",
        end_date="2024-12-31",
        output_path="data/spy.csv",
    )

    download_eod_data(
        symbol="EURUSD.FOREX",
        start_date="2010-01-01",
        end_date="2024-12-31",
        output_path="data/eurusd.csv",
    )

    download_eod_data(
        symbol="XAUUSD.FOREX",
        start_date="2010-01-01",
        end_date="2024-12-31",
        output_path="data/xauusd.csv",
    )