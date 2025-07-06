import pandas as pd
from datetime import date

def build_sleep_df(
    night: date,
    total_h: float,
    rem_min: int,
    deep_min: int,
    light_min: int,
    hrv_ms: int | None = None,
) -> pd.DataFrame:
    """Zamienia ręczny wpis snu w DataFrame (1 wiersz)."""
    return pd.DataFrame(
        {
            "date": [pd.to_datetime(night)],
            "total_sleep_h": [total_h],
            "rem_min": [rem_min],
            "deep_min": [deep_min],
            "light_min": [light_min],
            "hrv": [hrv_ms],
        }
    )
