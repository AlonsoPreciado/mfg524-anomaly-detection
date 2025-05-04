from data_loader import iter_batches
from preprocess  import preprocess
from detector    import ZScoreDetector, IsoForestDetector

def test_detectors():
    raw   = next(iter_batches(batch_size=800))
    clean = preprocess(raw)

    zdet = ZScoreDetector(threshold=3.0)
    assert zdet.predict(clean).dtype == bool

    idet = IsoForestDetector(contamination=0.05, random_state=0)
    idet.fit(clean)
    preds = idet.predict(clean)
    assert preds.dtype == bool and len(preds) == len(clean)
