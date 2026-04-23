from __future__ import annotations

from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {"date", "close"}


def load_price_data(path: str) -> pd.DataFrame:
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Plik nie istnieje: {path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Brak wymaganych kolumn {sorted(missing)} w pliku: {path}. "
            f"Dostępne kolumny: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError(f"Pusty plik wejściowy: {path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["date", "close"]).copy()
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"Po czyszczeniu danych nie zostały żadne poprawne wiersze: {path}")

    return df[["date", "close"]].copy()


if __name__ == "__main__":
    df = load_price_data("data/spy.csv")
    print(df.head())