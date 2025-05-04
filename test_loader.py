from data_loader import iter_batches

def test_iter_batches_sizes():
    sizes = [len(df) for df in iter_batches(batch_size=1200)]
    assert all(s <= 1200 for s in sizes)
    assert sum(sizes) > 0          # collection not empty
