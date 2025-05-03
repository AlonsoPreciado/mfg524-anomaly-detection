from synthetic_data_generator import make_dataset

def test_shape_and_flags():
    df = make_dataset(1000, anomaly_rate=0.05, seed=123)
    assert df.shape == (1000, 9)
    assert df["is_anomaly"].sum() == 50
