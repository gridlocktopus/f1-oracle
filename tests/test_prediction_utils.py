import numpy as np

from f1_oracle.predict.utils import sample_plackett_luce, rankings_to_position_probs


def test_rankings_to_position_probs_sums_to_one() -> None:
    scores = np.array([1.0, 2.0, 3.0])
    rankings = sample_plackett_luce(scores, n_samples=200, temperature=1.0)
    probs = rankings_to_position_probs(rankings)

    # Each driver's position distribution should sum to ~1
    row_sums = probs.sum(axis=1)
    for s in row_sums:
        assert abs(float(s) - 1.0) < 1e-6
