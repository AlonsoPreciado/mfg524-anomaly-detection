from data_loader import iter_batches
from preprocess import preprocess

def test_rolling_features_exist():
    df = preprocess(next(iter_batches(500)))
    assert "accel_x_roll_mean" in df.columns
    assert "gyro_z_roll_std"   in df.columns
    # ensure rows were dropped (rolling window)
    assert len(df) < 500
