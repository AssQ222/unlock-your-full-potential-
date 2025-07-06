from src.data_loader import build_sleep_df
from datetime import date

def test_build_sleep_df():
    df = build_sleep_df(date.today(), 7.5, 90, 60, 250, 75)
    assert df.shape == (1, 6)
    assert df["total_sleep_h"].iloc[0] == 7.5
